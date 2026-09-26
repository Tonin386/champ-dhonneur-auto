//! Matchs entre agents à recherche neuronale (évaluation, tournois), avec le moteur Rust.
//!
//! Chaque partie est donnée par sa graine et l'agent de chaque camp : un match apparié joue
//! chaque graine deux fois en inversant les agents (mêmes armées, ou mêmes 8 cartes de draft et
//! même premier à choisir, mêmes pioches). Toutes les parties d'un tournoi peuvent avancer
//! ensemble : les requêtes de chaque recherche vont à l'agent qui la mène, et [`Match::etape`]
//! renvoie un lot par agent, que Python fait évaluer par le réseau de cet agent (voir
//! `champ_dhonneur/ia/rs.py`). Les parties sont lancées au fil de l'eau (`simultanees` à la
//! fois) : les lots restent pleins, et on peut lire les résultats et arrêter en cours de route.

use crate::autojeu::{assembler_lot, Lot};
use crate::moteur::{Action, Partie};
use crate::draft::Moteur;
use crate::recherche::{ParamsRecherche, Requete};
use crate::rng::Rapide;

#[derive(Clone)]
pub struct ParamsMatch {
    /// parties : (graine, agent du camp 0, agent du camp 1)
    pub parties: Vec<(u64, usize, usize)>,
    /// mise en place avancée (draft) ; sinon armées tirées au hasard par la graine
    pub draft: bool,
    pub max_manches: u16,
    pub simultanees: usize,
    /// paramètres de recherche de chaque agent
    pub agents: Vec<ParamsRecherche>,
    /// parties dont le déroulé est renvoyé (relevés)
    pub releves: usize,
    /// fils d'exécution pour avancer les parties (le résultat n'en dépend pas)
    pub fils: usize,
}

/// Résultat d'une partie.
pub struct ResultatPartie {
    pub graine: u64,
    pub agents: [usize; 2],
    /// du point de vue du camp 0 : 1 victoire, 0 nulle, -1 défaite
    pub score0: i8,
    pub manches: u16,
    pub decisions: u32,
    pub actions: Option<Vec<Action>>,
    pub resultat: &'static str,
}

struct Table {
    jeu: Partie,
    graine: u64,
    agents: [usize; 2],
    rng: Rapide,
    recherche: Option<Moteur>,
    /// agent qui mène la recherche en cours
    agent: usize,
    requetes: Vec<Requete>,
    journal: Vec<Action>,
    decisions: u32,
    finie: bool,
}

impl Table {
    /// Transmet les réponses du réseau à la recherche en cours (et joue son coup si elle est
    /// terminée), puis lance la recherche suivante si besoin.
    fn avancer(&mut self, reps: &[(&[f32], f32)], agents: &[ParamsRecherche], legal: &mut Vec<Action>) {
        if !reps.is_empty() {
            let suite = self.recherche.as_mut().unwrap().repondre(reps);
            if suite.is_empty() {
                let res = self.recherche.take().unwrap().prendre_resultat().unwrap();
                self.jeu.jouer(&res.action).unwrap();
                self.journal.push(res.action);
                self.decisions += 1;
            } else {
                self.requetes = suite;
            }
        }
        if self.recherche.is_none() && !self.finie {
            self.finie = !self.preparer(agents, legal);
        }
    }

    /// Avance la partie jusqu'à sa prochaine vraie décision et lance la recherche de l'agent au
    /// trait. Renvoie faux si la partie est terminée.
    fn preparer(&mut self, agents: &[ParamsRecherche], legal: &mut Vec<Action>) -> bool {
        loop {
            if self.jeu.e.fini {
                return false;
            }
            self.jeu.actions_legales(legal);
            if legal.len() == 1 {
                self.jeu.jouer(&legal[0]).unwrap();
                self.journal.push(legal[0]);
                continue;
            }
            let agent = self.agents[self.jeu.au_trait() as usize];
            let params = agents[agent];
            let (r, reqs) = Moteur::nouvelle(&self.jeu, params.simulations, params, self.rng.next_u64());
            self.agent = agent;
            self.recherche = Some(r);
            self.requetes = reqs;
            return true;
        }
    }
}

/// Avance toutes les tables, réparties sur `fils` fils d'exécution (au moins 24 tables chacun).
fn avancer_tables(tables: &mut [Table], reps: &[Vec<(&[f32], f32)>], agents: &[ParamsRecherche], fils: usize) {
    let n = tables.len();
    let fils = fils.min(n.div_ceil(24)).max(1);
    if fils == 1 {
        let mut legal = Vec::with_capacity(48);
        for (t, r) in tables.iter_mut().zip(reps) {
            t.avancer(r, agents, &mut legal);
        }
        return;
    }
    let taille = n.div_ceil(fils);
    std::thread::scope(|s| {
        for (ts, rs) in tables.chunks_mut(taille).zip(reps.chunks(taille)) {
            s.spawn(move || {
                let mut legal = Vec::with_capacity(48);
                for (t, r) in ts.iter_mut().zip(rs) {
                    t.avancer(r, agents, &mut legal);
                }
            });
        }
    });
}

pub struct Match {
    p: ParamsMatch,
    a_lancer: Vec<(u64, usize, usize)>,
    lancees: usize,
    tables: Vec<Table>,
    lots_courants: Vec<Vec<(usize, usize)>>, // par agent : (table, nombre de requêtes)
    pub resultats: Vec<ResultatPartie>,
    pub evaluations: Vec<u64>,
}

impl Match {
    pub fn new(p: ParamsMatch) -> Match {
        let n = p.agents.len();
        let a_lancer = p.parties.iter().rev().cloned().collect();
        Match {
            p,
            a_lancer,
            lancees: 0,
            tables: Vec::new(),
            lots_courants: vec![Vec::new(); n],
            resultats: Vec::new(),
            evaluations: vec![0; n],
        }
    }

    /// Transmet les réponses aux lots précédents (une par agent, None si son lot était vide) et
    /// renvoie les lots suivants (un par agent, éventuellement vide), ou `None` quand toutes les
    /// parties sont terminées.
    pub fn etape(&mut self, reponses: &[Option<(&[f32], usize, &[f32])>]) -> Option<Vec<Lot>> {
        let mut par_table: Vec<Vec<(&[f32], f32)>> = (0..self.tables.len()).map(|_| Vec::new()).collect();
        for (agent, rep) in reponses.iter().enumerate() {
            let lot = std::mem::take(&mut self.lots_courants[agent]);
            let Some((logits, a_len, valeurs)) = *rep else {
                assert!(lot.is_empty(), "réponses manquantes pour l'agent {agent}");
                continue;
            };
            let mut k = 0;
            for (t, n) in lot {
                par_table[t] = (k..k + n).map(|i| (&logits[i * a_len..(i + 1) * a_len], valeurs[i])).collect();
                k += n;
                self.evaluations[agent] += n as u64;
            }
        }
        loop {
            while self.tables.len() < self.p.simultanees {
                let Some((graine, a0, a1)) = self.a_lancer.pop() else { break };
                let jeu = if self.p.draft {
                    Partie::nouvelle_draft(graine, None, None, self.p.max_manches)
                } else {
                    Partie::nouvelle(graine, None, None, self.p.max_manches)
                };
                self.lancees += 1;
                self.tables.push(Table {
                    jeu,
                    graine,
                    agents: [a0, a1],
                    // recherches reproductibles : ne dépendent que de la partie (pas de l'ordre)
                    rng: Rapide::new(graine.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((a0 as u64) << 32 | a1 as u64)),
                    recherche: None,
                    agent: 0,
                    requetes: Vec::new(),
                    journal: Vec::new(),
                    decisions: 0,
                    finie: false,
                });
                par_table.push(Vec::new());
            }
            avancer_tables(&mut self.tables, &par_table, &self.p.agents, self.p.fils);
            par_table.iter_mut().for_each(|v| v.clear());
            let mut i = 0;
            while i < self.tables.len() {
                if self.tables[i].finie {
                    let t = self.tables.swap_remove(i);
                    par_table.swap_remove(i);
                    self.terminer(t);
                    continue;
                }
                i += 1;
            }
            if self.tables.is_empty() {
                return None;
            }
            if self.tables.iter().any(|t| !t.requetes.is_empty()) {
                return Some((0..self.p.agents.len()).map(|a| self.assembler(a)).collect());
            }
        }
    }

    /// Nombre de parties lancées (terminées ou en cours).
    pub fn lancees(&self) -> usize {
        self.lancees
    }

    fn terminer(&mut self, t: Table) {
        let g = &t.jeu;
        let score0 = if g.e.gagnant < 0 {
            0
        } else if g.e.gagnant as u8 == g.equipe(0) {
            1
        } else {
            -1
        };
        // les premières parties terminées gardent leur déroulé (résultats lus au fil de l'eau)
        let garder = self.p.releves > 0;
        self.p.releves -= garder as usize;
        self.resultats.push(ResultatPartie {
            graine: t.graine,
            agents: t.agents,
            score0,
            manches: g.e.manche,
            decisions: t.decisions,
            actions: if garder { Some(t.journal) } else { None },
            resultat: g.resultat(),
        });
    }

    fn assembler(&mut self, agent: usize) -> Lot {
        let mut courant = Vec::new();
        let mut groupes = Vec::new();
        for (i, t) in self.tables.iter_mut().enumerate() {
            if t.agent == agent && !t.requetes.is_empty() {
                courant.push((i, t.requetes.len()));
                groupes.push(std::mem::take(&mut t.requetes));
            }
        }
        self.lots_courants[agent] = courant;
        assembler_lot(groupes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jouer(p: ParamsMatch) -> Vec<ResultatPartie> {
        let n = p.agents.len();
        let mut m = Match::new(p);
        let mut rep: Vec<Option<(Vec<f32>, usize, Vec<f32>)>> = vec![None; n];
        let mut out = Vec::new();
        loop {
            let r: Vec<Option<(&[f32], usize, &[f32])>> =
                rep.iter().map(|x| x.as_ref().map(|(l, a, v)| (l.as_slice(), *a, v.as_slice()))).collect();
            let Some(lots) = m.etape(&r) else { break };
            out.append(&mut m.resultats); // lecture en cours de route
            rep = lots
                .iter()
                .map(|l| (l.b > 0).then(|| (vec![0.0; l.b * l.a_len], l.a_len, vec![0.0; l.b])))
                .collect();
        }
        out.append(&mut m.resultats);
        out
    }

    /// Chaque graine est jouée avec les agents dans les deux camps ; tout est reproductible,
    /// quel que soit le nombre de parties simultanées.
    #[test]
    fn match_apparie_et_reproductible() {
        let agent = ParamsRecherche { simulations: 8, bruit: false, bruit_coup: false, ..Default::default() };
        let agents = vec![agent, ParamsRecherche { simulations: 4, ..agent }, ParamsRecherche { simulations: 6, ..agent }];
        let mut parties = Vec::new();
        for g in 0..20u64 {
            parties.extend([(g, 0, 1), (g, 1, 0), (g, 1, 2), (g, 2, 1)]);
        }
        let p = ParamsMatch { parties, draft: true, max_manches: 30, simultanees: 5, agents, releves: 2, fils: 1 };
        let r1 = jouer(p.clone());
        let r2 = jouer(ParamsMatch { simultanees: 80, ..p.clone() });
        let r3 = jouer(ParamsMatch { simultanees: 80, fils: 4, ..p }); // 4 fils de 20 parties
        assert_eq!(r1.len(), 80);
        let cle = |r: &Vec<ResultatPartie>| {
            let mut v: Vec<(u64, [usize; 2], i8, u16)> = r.iter().map(|x| (x.graine, x.agents, x.score0, x.manches)).collect();
            v.sort();
            v
        };
        assert_eq!(cle(&r1), cle(&r2));
        assert_eq!(cle(&r1), cle(&r3));
        assert_eq!(r1.iter().filter(|r| r.actions.is_some()).count(), 2);
    }
}
