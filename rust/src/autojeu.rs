//! Auto-jeu : portage de `ia/autojeu.py`. Des dizaines de parties avancent ensemble ; leurs
//! requêtes forment un lot unique que Python fait évaluer par le réseau (GPU).

use crate::encodage::*;
use crate::moteur::{Action, Partie, N_TYPES};
use crate::plateau::{plateau, AUCUNE_CASE, N_CASES};
use crate::recherche::{ParamsRecherche, Recherche, Requete};
use crate::rng::Rapide;

pub const A_MAX: usize = 64;

#[derive(Clone, Copy)]
pub struct ParamsAutoJeu {
    pub parties: usize,
    pub simultanees: usize,
    pub simulations: usize,
    pub simulations_rapides: usize,
    pub p_complete: f64,
    pub max_manches: u16,
    pub m: usize,
    pub parallele: usize,
    pub c_visit: f64,
    pub c_scale: f64,
    pub releves: usize,
    /// part des parties commencées par la mise en place avancée (draft)
    pub p_draft: f64,
    /// simulations aux décisions de draft (toujours des recherches complètes)
    pub simulations_draft: usize,
    /// coup joué = meilleur coup de la recherche (sans bruit), coup gagnant toujours joué ; le bruit
    /// de Gumbel ne sert plus qu'à choisir les candidats des recherches complètes (cibles de
    /// politique), les recherches rapides n'en ont pas ; la valeur de recherche des exemples est
    /// celle du coup joué
    pub meilleur_coup: bool,
}

struct Exemple {
    obs: Observation,
    acts: Vec<i64>,
    pi: Vec<f32>,
    q: f32,
    equipe: u8,
    /// main réelle de l'adversaire (cible auxiliaire de croyance), par code de pièce
    main_adv: [i8; N_TYPES],
}

struct Emplacement {
    jeu: Partie,
    graine: u64,
    exemples: Vec<Exemple>,
    recherche: Option<Recherche>,
    complet: bool,
    journal: Vec<Action>,
    requetes: Vec<Requete>,
}

#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub parties: u64,
    pub victoires_blanc: u64,
    pub victoires_noir: u64,
    pub nulles: u64,
    pub manches: u64,
    pub decisions: u64,
    pub recherches: u64,
    pub evaluations: u64,
    pub parties_draft: u64,
    /// parties avec draft gagnées par le premier à choisir (A)
    pub victoires_choisit: u64,
    /// parties gagnées par le premier joueur de la manche 1
    pub victoires_premier: u64,
}

/// Exemples empaquetés (mêmes champs que `empaqueter`, flottants en f32).
#[derive(Default)]
pub struct Donnees {
    pub n: usize,
    pub cell_i: Vec<i8>,
    pub cell_f: Vec<f32>,
    pub unit_i: Vec<i8>,
    pub unit_f: Vec<f32>,
    pub glob_i: Vec<i8>,
    pub glob_f: Vec<f32>,
    pub acts: Vec<i8>,
    pub pi: Vec<f32>,
    pub n_act: Vec<i16>,
    pub z: Vec<f32>,
    pub q: Vec<f32>,
    /// cibles auxiliaires : contrôle final de chaque case vue (-1 hors Lieu, 0 neutre, 1 à moi,
    /// 2 adverse), marge finale de marqueurs (adverses restants − miens), main adverse
    pub lieux: Vec<i8>,
    pub marge: Vec<i8>,
    pub main_adv: Vec<i8>,
}

/// Lot de positions à évaluer, à plat (entiers i64, flottants f32).
pub struct Lot {
    pub b: usize,
    pub a_len: usize,
    pub cell_i: Vec<i64>,
    pub cell_f: Vec<f32>,
    pub unit_i: Vec<i64>,
    pub unit_f: Vec<f32>,
    pub glob_i: Vec<i64>,
    pub glob_f: Vec<f32>,
    pub acts: Vec<i64>,
    pub n_act: Vec<i64>,
}

pub struct Releve {
    pub graine: u64,
    pub actions: Vec<Action>,
    pub resultat: &'static str,
    pub draft: bool,
}

/// Partie suivie pour la diffusion en direct (voir `ia/direct.py`).
pub struct Suivie {
    pub graine: u64,
    pub premier: u8,
    pub unites: [[u8; 4]; 2],
    pub actions: Vec<Action>,
    pub fini: bool,
    pub draft: bool,
}

fn instantane(e: &Emplacement, fini: bool) -> Suivie {
    Suivie {
        graine: e.graine,
        premier: e.jeu.e.premier,
        unites: [e.jeu.e.joueurs[0].unites, e.jeu.e.joueurs[1].unites],
        actions: e.journal.clone(),
        fini,
        draft: e.jeu.e.draft,
    }
}

pub struct AutoJeu {
    p: ParamsAutoJeu,
    rng: Rapide,
    emplacements: Vec<Emplacement>,
    lances: usize,
    lot_courant: Vec<(usize, usize)>, // (emplacement, nombre de requêtes)
    pub stats: Stats,
    pub donnees: Donnees,
    pub releves: Vec<Releve>,
    tampon: Vec<Action>,
    suivie: Option<u64>,       // graine de la partie suivie
    fin_suivie: Option<Suivie>, // position finale de la partie suivie, pas encore transmise
}

impl AutoJeu {
    pub fn new(p: ParamsAutoJeu, graine: u64) -> AutoJeu {
        AutoJeu {
            p,
            rng: Rapide::new(graine),
            emplacements: Vec::new(),
            lances: 0,
            lot_courant: Vec::new(),
            stats: Stats::default(),
            donnees: Donnees::default(),
            releves: Vec::new(),
            tampon: Vec::with_capacity(48),
            suivie: None,
            fin_suivie: None,
        }
    }

    fn params_recherche(&self, simulations: usize, complet: bool) -> ParamsRecherche {
        let meilleur = self.p.meilleur_coup;
        ParamsRecherche {
            simulations,
            m: self.p.m,
            c_visit: self.p.c_visit,
            c_scale: self.p.c_scale,
            bruit: complet || !meilleur,
            parallele: self.p.parallele,
            bruit_coup: !meilleur,
            coup_gagnant: meilleur,
            ..ParamsRecherche::default()
        }
    }

    /// Transmet les réponses du lot précédent (s'il y en a un) et renvoie le lot suivant,
    /// ou `None` quand toutes les parties sont terminées.
    pub fn etape(&mut self, reponses: Option<(&[f32], usize, &[f32])>) -> Option<Lot> {
        if let Some((logits, a_len, valeurs)) = reponses {
            let lot = std::mem::take(&mut self.lot_courant);
            let mut k = 0;
            for (e, n) in lot {
                let reps: Vec<(&[f32], f32)> =
                    (k..k + n).map(|i| (&logits[i * a_len..(i + 1) * a_len], valeurs[i])).collect();
                k += n;
                self.stats.evaluations += n as u64;
                let suite = self.emplacements[e].recherche.as_mut().unwrap().repondre(&reps);
                if suite.is_empty() {
                    self.fin_recherche(e);
                } else {
                    self.emplacements[e].requetes = suite;
                }
            }
        }
        loop {
            while self.emplacements.len() < self.p.simultanees && self.lances < self.p.parties {
                let graine = self.rng.next_u64() % (1u64 << 31);
                let draft = self.p.p_draft > 0.0 && self.rng.uniforme() < self.p.p_draft;
                let jeu = if draft {
                    Partie::nouvelle_draft(graine, None, None, self.p.max_manches)
                } else {
                    Partie::nouvelle(graine, None, None, self.p.max_manches)
                };
                self.emplacements.push(Emplacement {
                    jeu,
                    graine,
                    exemples: Vec::new(),
                    recherche: None,
                    complet: false,
                    journal: Vec::new(),
                    requetes: Vec::new(),
                });
                self.lances += 1;
            }
            let mut i = 0;
            while i < self.emplacements.len() {
                if self.emplacements[i].recherche.is_none() && !self.preparer(i) {
                    let e = self.emplacements.swap_remove(i);
                    self.terminer(e);
                    continue;
                }
                i += 1;
            }
            if self.emplacements.is_empty() {
                return None;
            }
            if self.emplacements.iter().any(|e| !e.requetes.is_empty()) {
                return Some(self.assembler());
            }
        }
    }

    /// Avance la partie jusqu'à sa prochaine vraie décision et lance la recherche.
    /// Renvoie faux si la partie est terminée.
    fn preparer(&mut self, i: usize) -> bool {
        let mut legal = std::mem::take(&mut self.tampon);
        let ok = loop {
            let e = &mut self.emplacements[i];
            if e.jeu.e.fini {
                break false;
            }
            e.jeu.actions_legales(&mut legal);
            if legal.len() == 1 {
                e.jeu.jouer(&legal[0]).unwrap();
                e.journal.push(legal[0]);
                self.stats.decisions += 1;
                continue;
            }
            let en_draft = e.jeu.e.en_draft;
            let complet = en_draft || self.rng.uniforme() < self.p.p_complete;
            let sims = if en_draft {
                self.p.simulations_draft
            } else if complet {
                self.p.simulations
            } else {
                self.p.simulations_rapides
            };
            let params = self.params_recherche(sims, complet);
            let graine = self.rng.next_u64();
            let e = &mut self.emplacements[i];
            let (r, reqs) = Recherche::nouvelle(&e.jeu, sims, params, graine);
            e.complet = complet;
            e.recherche = Some(r);
            e.requetes = reqs;
            break true;
        };
        self.tampon = legal;
        ok
    }

    fn fin_recherche(&mut self, i: usize) {
        let e = &mut self.emplacements[i];
        let res = e.recherche.take().unwrap().resultat.unwrap();
        if e.complet {
            // valeur de recherche : celle du coup joué (meilleur_coup), sinon la moyenne de la racine
            let q = if self.p.meilleur_coup {
                res.q[res.legal.iter().position(|a| *a == res.action).unwrap()]
            } else {
                res.valeur
            };
            let mut acts = Vec::with_capacity(res.legal.len() * ACT_F);
            encoder_actions(&e.jeu, &res.legal, &mut acts);
            let adv = &e.jeu.e.joueurs[1 - e.jeu.au_trait() as usize];
            let mut main_adv = [0i8; N_TYPES];
            for &c in adv.main.as_slice() {
                main_adv[c as usize] += 1;
            }
            e.exemples.push(Exemple {
                main_adv,
                obs: encoder_etat(&e.jeu),
                acts,
                pi: res.politique.clone(),
                q: q as f32,
                equipe: e.jeu.equipe(e.jeu.au_trait()),
            });
        }
        e.jeu.jouer(&res.action).unwrap();
        e.journal.push(res.action);
        self.stats.decisions += 1;
        self.stats.recherches += 1;
    }

    /// Partie suivie pour la diffusion en direct : sa position finale une fois (fini = vrai),
    /// puis une autre partie en cours.
    pub fn suivie(&mut self) -> Option<Suivie> {
        if let Some(f) = self.fin_suivie.take() {
            return Some(f);
        }
        if self.suivie.is_none() {
            self.suivie = self.emplacements.first().map(|e| e.graine);
        }
        let g = self.suivie?;
        self.emplacements.iter().find(|e| e.graine == g).map(|e| instantane(e, false))
    }

    fn terminer(&mut self, e: Emplacement) {
        if self.suivie == Some(e.graine) {
            self.fin_suivie = Some(instantane(&e, true));
            self.suivie = None;
        }
        let g = &e.jeu;
        let s = &mut self.stats;
        s.parties += 1;
        s.manches += g.e.manche as u64;
        if g.e.gagnant >= 0 && g.e.gagnant as u8 == g.equipe(g.e.premier) {
            s.victoires_premier += 1;
        }
        if g.e.draft {
            s.parties_draft += 1;
            if g.e.gagnant >= 0 && g.e.gagnant as u8 == g.equipe(g.e.choisit) {
                s.victoires_choisit += 1;
            }
        }
        match g.e.gagnant {
            -1 => s.nulles += 1,
            0 => s.victoires_blanc += 1,
            _ => s.victoires_noir += 1,
        }
        let pl = plateau();
        for ex in &e.exemples {
            let z = if g.e.gagnant < 0 { 0.0 } else if g.e.gagnant as u8 == ex.equipe { 1.0 } else { -1.0 };
            let t = ex.equipe;
            let mut lieux = [-1i8; N_CASES];
            for &l in &pl.lieux {
                let v = if t == 0 { l as usize } else { pl.rotation[l as usize] as usize };
                let c = g.e.controle[l as usize];
                lieux[v] = if c < 0 { 0 } else if c as u8 == t { 1 } else { 2 };
            }
            let marge = (g.e.marqueurs[1 - t as usize] - g.e.marqueurs[t as usize]).clamp(-4, 4);
            self.empaqueter(ex, z, &lieux, marge);
        }
        // relevés tirés au hasard (réservoir) : les premières parties finies sont les plus courtes
        let r = Releve { graine: e.graine, actions: e.journal.clone(), resultat: g.resultat(), draft: g.e.draft };
        if self.releves.len() < self.p.releves {
            self.releves.push(r);
        } else if self.p.releves > 0 {
            let k = (self.rng.next_u64() % self.stats.parties) as usize;
            if k < self.p.releves {
                self.releves[k] = r;
            }
        }
    }

    fn empaqueter(&mut self, ex: &Exemple, z: f32, lieux: &[i8; N_CASES], marge: i8) {
        let d = &mut self.donnees;
        d.n += 1;
        d.lieux.extend_from_slice(lieux);
        d.marge.push(marge);
        d.main_adv.extend_from_slice(&ex.main_adv);
        d.cell_i.extend(ex.obs.cell_i.iter().map(|&v| v as i8));
        d.cell_f.extend_from_slice(&ex.obs.cell_f);
        d.unit_i.extend(ex.obs.unit_i.iter().map(|&v| v as i8));
        d.unit_f.extend_from_slice(&ex.obs.unit_f);
        d.glob_i.extend(ex.obs.glob_i.iter().map(|&v| v as i8));
        d.glob_f.extend_from_slice(&ex.obs.glob_f);
        let n = ex.pi.len();
        // très rare : plus de A_MAX actions, on garde les plus probables
        let mut garde: Vec<usize> = (0..n).collect();
        if n > A_MAX {
            garde.sort_by(|&a, &b| ex.pi[b].partial_cmp(&ex.pi[a]).unwrap());
            garde.truncate(A_MAX);
        }
        let somme: f32 = garde.iter().map(|&i| ex.pi[i]).sum();
        for k in 0..A_MAX {
            if k < garde.len() {
                let i = garde[k];
                d.acts.extend(ex.acts[i * ACT_F..(i + 1) * ACT_F].iter().map(|&v| v as i8));
                d.pi.push(if n > A_MAX { ex.pi[i] / somme } else { ex.pi[i] });
            } else {
                d.acts.extend_from_slice(&[0, 0, 0, 0, AUCUNE_CASE as i8, AUCUNE_CASE as i8, AUCUNE_CASE as i8]);
                d.pi.push(0.0);
            }
        }
        d.n_act.push(garde.len() as i16);
        d.z.push(z);
        d.q.push(ex.q);
    }

    fn assembler(&mut self) -> Lot {
        let mut lot_courant = Vec::new();
        let mut b = 0;
        let mut a_len = 1;
        for (i, e) in self.emplacements.iter().enumerate() {
            if !e.requetes.is_empty() {
                lot_courant.push((i, e.requetes.len()));
                b += e.requetes.len();
                for r in &e.requetes {
                    a_len = a_len.max(r.n_act);
                }
            }
        }
        let mut lot = Lot {
            b,
            a_len,
            cell_i: Vec::with_capacity(b * N_CASES * CELL_I),
            cell_f: Vec::with_capacity(b * N_CASES * CELL_F),
            unit_i: Vec::with_capacity(b * N_UNITES * UNIT_I),
            unit_f: Vec::with_capacity(b * N_UNITES * UNIT_F),
            glob_i: Vec::with_capacity(b * GLOB_I),
            glob_f: Vec::with_capacity(b * GLOB_F),
            acts: Vec::with_capacity(b * a_len * ACT_F),
            n_act: Vec::with_capacity(b),
        };
        for &(i, _) in &lot_courant {
            for r in std::mem::take(&mut self.emplacements[i].requetes) {
                lot.cell_i.extend_from_slice(&r.obs.cell_i);
                lot.cell_f.extend_from_slice(&r.obs.cell_f);
                lot.unit_i.extend_from_slice(&r.obs.unit_i);
                lot.unit_f.extend_from_slice(&r.obs.unit_f);
                lot.glob_i.extend_from_slice(&r.obs.glob_i);
                lot.glob_f.extend_from_slice(&r.obs.glob_f);
                lot.acts.extend_from_slice(&r.actions);
                for _ in r.n_act..a_len {
                    lot.acts.extend_from_slice(&[0, 0, 0, 0, AUCUNE_CASE as i64, AUCUNE_CASE as i64, AUCUNE_CASE as i64]);
                }
                lot.n_act.push(r.n_act as i64);
            }
        }
        self.lot_courant = lot_courant;
        lot
    }
}
