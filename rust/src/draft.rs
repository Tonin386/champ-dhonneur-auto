//! Choix exact des cartes pendant la mise en place avancée (draft).
//!
//! Au lieu d'une recherche arborescente à chaque choix de carte, toutes les répartitions finales
//! des cartes restantes (au plus 70 au premier choix) sont évaluées par le réseau : pour chacune,
//! le draft est achevé, la manche 1 commence (pioches) et la position de la première vraie
//! décision est évaluée, en moyenne sur `k` tirages des sacs. Le choix de chaque carte est
//! ensuite exact par minimax sur l'ordre des choix (A1 B2 A2 B2 A1). Toutes les évaluations
//! partent en un seul lot (une étape au lieu de dizaines de simulations successives).
//!
//! Cible de politique : même transformation que la recherche Gumbel, softmax(logits + σ(Q)),
//! avec les Q exacts du minimax et `k` comme nombre de visites.

use std::collections::HashMap;

use crate::moteur::{Action, Partie, N_CARTES, ORDRE_DRAFT};
use crate::recherche::{requete, ParamsRecherche, Recherche, Requete, Resultat};
use crate::rng::Rapide;

/// Recherche d'une décision : Gumbel IS-MCTS, ou choix exact de carte pendant le draft
/// (`ParamsRecherche::draft_exact` > 0 : nombre de tirages par répartition).
pub enum Moteur {
    Gumbel(Recherche),
    Draft(RechercheDraft),
}

impl Moteur {
    pub fn nouvelle(jeu: &Partie, simulations: usize, p: ParamsRecherche, graine: u64) -> (Moteur, Vec<Requete>) {
        if jeu.e.en_draft && p.draft_exact > 0 {
            let (r, q) = RechercheDraft::nouvelle(jeu, p, p.draft_exact, graine);
            (Moteur::Draft(r), q)
        } else {
            let (r, q) = Recherche::nouvelle(jeu, simulations, p, graine);
            (Moteur::Gumbel(r), q)
        }
    }

    pub fn repondre(&mut self, reponses: &[(&[f32], f32)]) -> Vec<Requete> {
        match self {
            Moteur::Gumbel(r) => r.repondre(reponses),
            Moteur::Draft(r) => r.repondre(reponses),
        }
    }

    /// Arrêt demandé : dernières réponses, puis résultat sans autre simulation.
    pub fn repondre_et_terminer(&mut self, reponses: &[(&[f32], f32)]) {
        match self {
            Moteur::Gumbel(r) => r.repondre_et_terminer(reponses),
            Moteur::Draft(r) => {
                r.repondre(reponses);
            }
        }
    }

    pub fn resultat(&self) -> Option<&Resultat> {
        match self {
            Moteur::Gumbel(r) => r.resultat.as_ref(),
            Moteur::Draft(r) => r.resultat.as_ref(),
        }
    }

    /// Recherche progressive (Gumbel seulement ; le draft exact est déjà complet).
    pub fn prolonger(&mut self, budget: usize, candidats: usize) -> Vec<Requete> {
        match self {
            Moteur::Gumbel(r) => r.prolonger(budget, candidats),
            Moteur::Draft(_) => Vec::new(),
        }
    }

    pub fn lignes(&self, profondeur: usize) -> Vec<Vec<(crate::moteur::Action, f64)>> {
        match self {
            Moteur::Gumbel(r) => r.lignes(profondeur),
            Moteur::Draft(r) => vec![Vec::new(); r.nombre_coups()],
        }
    }

    pub fn prendre_resultat(self) -> Option<Resultat> {
        match self {
            Moteur::Gumbel(r) => r.resultat,
            Moteur::Draft(r) => r.resultat,
        }
    }
}

pub struct RechercheDraft {
    p: ParamsRecherche,
    equipe: u8,
    legal: Vec<Action>,
    /// cartes déjà prises par chaque joueur et cartes restantes (bit u-1 : carte u)
    base: [u16; 2],
    dispo: u16,
    etape: usize,
    choisit: u8,
    equipes: [u8; 2],
    feuilles: HashMap<[u16; 2], usize>,
    valeurs: Vec<f64>,
    n_valeurs: Vec<u32>,
    /// (feuille, signe) de chaque requête après celle de la racine
    attente: Vec<(usize, f64)>,
    pub resultat: Option<Resultat>,
}

fn cartes(m: u16) -> impl Iterator<Item = u8> {
    (1..=16u8).filter(move |&u| m & (1 << (u - 1)) != 0)
}

impl RechercheDraft {
    /// `jeu` : une décision de draft. Renvoie la recherche et toutes ses requêtes (racine, puis
    /// chaque répartition finale × `k` tirages).
    pub fn nouvelle(jeu: &Partie, p: ParamsRecherche, k: usize, graine: u64) -> (RechercheDraft, Vec<Requete>) {
        let e = &jeu.e;
        let moi = jeu.au_trait();
        let mut legal = Vec::new();
        jeu.actions_legales(&mut legal);
        let base = [0, 1].map(|j: usize| e.joueurs[j].unites().iter().fold(0u16, |m, &u| m | 1 << (u - 1)));
        let etape = e.etape_draft as usize;
        let choisit = e.choisit;
        let restants: [usize; 2] = [0, 1].map(|j| {
            (etape..N_CARTES).filter(|&s| (choisit + ORDRE_DRAFT[s]) % 2 == j as u8).count()
        });
        let dispo: Vec<u8> = cartes(e.dispo).collect();
        // répartitions finales : les cartes restantes du joueur 0, parmi les disponibles
        let mut feuilles = HashMap::new();
        let n = dispo.len();
        for sous in 0u32..(1 << n) {
            if sous.count_ones() as usize != restants[0] {
                continue;
            }
            let mut m0 = base[0];
            let mut m1 = base[1];
            for (i, &u) in dispo.iter().enumerate() {
                if sous & (1 << i) != 0 {
                    m0 |= 1 << (u - 1);
                } else {
                    m1 |= 1 << (u - 1);
                }
            }
            let id = feuilles.len();
            feuilles.insert([m0, m1], id);
        }
        let mut rng = Rapide::new(graine);
        let equipes = [jeu.equipe(0), jeu.equipe(1)];
        let equipe = equipes[moi as usize];
        let mut reqs = vec![requete(jeu, &legal)];
        let mut attente = Vec::with_capacity(feuilles.len() * k);
        let mut tampon = Vec::new();
        let mut ordre: Vec<(&[u16; 2], &usize)> = feuilles.iter().collect();
        ordre.sort_by_key(|(_, &id)| id);
        for (&parts, &id) in ordre {
            for _ in 0..k.max(1) {
                let mut g = jeu.copie_rapide(rng.next_u64());
                // achever le draft : chacun prend ses cartes dans l'ordre croissant
                while g.e.en_draft {
                    let j = g.au_trait() as usize;
                    let tenues = g.e.joueurs[j].unites().iter().fold(0u16, |m, &u| m | 1 << (u - 1));
                    let u = cartes(parts[j] & !tenues).next().expect("répartition incohérente");
                    g.actions_legales(&mut tampon);
                    let a = *tampon.iter().find(|a| a.unite == u).expect("carte indisponible");
                    g.jouer(&a).unwrap();
                }
                // jusqu'à la première vraie décision (pioches de la manche 1 faites)
                loop {
                    g.actions_legales(&mut tampon);
                    if g.e.fini || tampon.len() != 1 {
                        break;
                    }
                    let a = tampon[0];
                    g.jouer(&a).unwrap();
                }
                if g.e.fini {
                    continue;
                }
                let signe = if g.equipe(g.au_trait()) == equipe { 1.0 } else { -1.0 };
                reqs.push(requete(&g, &tampon));
                attente.push((id, signe));
            }
        }
        let nf = feuilles.len();
        let r = RechercheDraft {
            p,
            equipe,
            legal,
            base,
            dispo: e.dispo,
            etape,
            choisit,
            equipes,
            feuilles,
            valeurs: vec![0.0; nf],
            n_valeurs: vec![0; nf],
            attente,
            resultat: None,
        };
        (r, reqs)
    }

    pub fn nombre_coups(&self) -> usize {
        self.legal.len()
    }

    /// Réponses du réseau à toutes les requêtes (racine d'abord) : calcule le résultat.
    pub fn repondre(&mut self, reponses: &[(&[f32], f32)]) -> Vec<Requete> {
        let (logits_racine, v_racine) = reponses[0];
        for (&(id, signe), &(_, v)) in self.attente.iter().zip(&reponses[1..]) {
            self.valeurs[id] += signe * v as f64;
            self.n_valeurs[id] += 1;
        }
        let mut memo = HashMap::new();
        let q: Vec<f64> = self
            .legal
            .iter()
            .map(|a| {
                let j = (self.choisit + ORDRE_DRAFT[self.etape]) % 2;
                let mut m = self.base;
                m[j as usize] |= 1 << (a.unite - 1);
                self.minimax(m, self.etape + 1, &mut memo)
            })
            .collect();
        let k = self.legal.len();
        let n_vis = (self.attente.len() as f64 / self.feuilles.len().max(1) as f64).max(1.0);
        let lo = q.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e = (hi - lo).max(1e-8);
        let z: Vec<f64> = (0..k)
            .map(|i| logits_racine[i] as f64 + (self.p.c_visit + n_vis) * self.p.c_scale * (q[i] - lo) / e)
            .collect();
        let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
        let s: f64 = ex.iter().sum();
        let politique: Vec<f32> = ex.iter().map(|v| (v / s) as f32).collect();
        let meilleur = (0..k).max_by(|&a, &b| q[a].partial_cmp(&q[b]).unwrap().then(b.cmp(&a))).unwrap();
        self.resultat = Some(Resultat {
            action: self.legal[meilleur],
            legal: self.legal.clone(),
            politique,
            visites: vec![n_vis; k],
            q: q.clone(),
            valeur: q[meilleur],
            v_reseau: v_racine as f64,
            simulations: self.attente.len(),
        });
        Vec::new()
    }

    /// Valeur (point de vue de l'équipe au trait à la racine) des cartes tenues `m` à l'étape `s`.
    fn minimax(&self, m: [u16; 2], s: usize, memo: &mut HashMap<[u16; 2], f64>) -> f64 {
        if s >= N_CARTES {
            let id = self.feuilles[&m];
            return if self.n_valeurs[id] > 0 { self.valeurs[id] / self.n_valeurs[id] as f64 } else { 0.0 };
        }
        if let Some(&v) = memo.get(&m) {
            return v;
        }
        let j = ((self.choisit + ORDRE_DRAFT[s]) % 2) as usize;
        let libres = self.dispo & !(m[0] | m[1]);
        let moi = self.equipes[j] == self.equipe;
        let mut best = if moi { f64::NEG_INFINITY } else { f64::INFINITY };
        for u in cartes(libres) {
            let mut m2 = m;
            m2[j] |= 1 << (u - 1);
            let v = self.minimax(m2, s + 1, memo);
            best = if moi { best.max(v) } else { best.min(v) };
        }
        memo.insert(m, best);
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Au premier choix : 70 répartitions × k tirages + la racine ; coup légal ; politique normée ;
    /// le minimax suit l'ordre des choix (une valeur ne change pas quand on permute les tirages).
    #[test]
    fn draft_exact() {
        let jeu = Partie::nouvelle_draft(4, None, None, 100);
        let p = ParamsRecherche { bruit: false, bruit_coup: false, ..Default::default() };
        let (mut r, reqs) = RechercheDraft::nouvelle(&jeu, p, 2, 7);
        assert_eq!(reqs.len(), 1 + 70 * 2);
        // valeurs factices, différentes d'une requête à l'autre
        let zeros: Vec<Vec<f32>> = reqs.iter().map(|q| vec![0.0; q.n_act]).collect();
        let reps: Vec<(&[f32], f32)> = zeros.iter().enumerate().map(|(i, z)| (z.as_slice(), if i % 3 == 0 { 0.3 } else { -0.1 })).collect();
        assert!(r.repondre(&reps).is_empty());
        let res = r.resultat.as_ref().unwrap();
        assert_eq!(res.legal.len(), 8);
        assert!(res.legal.contains(&res.action));
        assert!((res.politique.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(res.q.iter().all(|q| (-1.0..=1.0).contains(q)));
    }
}
