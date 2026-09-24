//! Auto-jeu : portage de `ia/autojeu.py`. Des dizaines de parties avancent ensemble ; leurs
//! requêtes forment un lot unique que Python fait évaluer par le réseau (GPU).

use crate::encodage::*;
use crate::moteur::{Action, Partie};
use crate::plateau::{AUCUNE_CASE, N_CASES};
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
}

struct Exemple {
    obs: Observation,
    acts: Vec<i64>,
    pi: Vec<f32>,
    q: f32,
    equipe: u8,
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
}

/// Partie suivie pour la diffusion en direct (voir `ia/direct.py`).
pub struct Suivie {
    pub graine: u64,
    pub premier: u8,
    pub unites: [[u8; 4]; 2],
    pub actions: Vec<Action>,
    pub fini: bool,
}

fn instantane(e: &Emplacement, fini: bool) -> Suivie {
    Suivie {
        graine: e.graine,
        premier: e.jeu.e.premier,
        unites: [e.jeu.e.joueurs[0].unites, e.jeu.e.joueurs[1].unites],
        actions: e.journal.clone(),
        fini,
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

    fn params_recherche(&self, simulations: usize) -> ParamsRecherche {
        ParamsRecherche {
            simulations,
            m: self.p.m,
            c_visit: self.p.c_visit,
            c_scale: self.p.c_scale,
            bruit: true,
            parallele: self.p.parallele,
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
                self.emplacements.push(Emplacement {
                    jeu: Partie::nouvelle(graine, None, None, self.p.max_manches),
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
            let complet = self.rng.uniforme() < self.p.p_complete;
            let sims = if complet { self.p.simulations } else { self.p.simulations_rapides };
            let params = self.params_recherche(sims);
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
            let mut acts = Vec::with_capacity(res.legal.len() * ACT_F);
            encoder_actions(&e.jeu, &res.legal, &mut acts);
            e.exemples.push(Exemple {
                obs: encoder_etat(&e.jeu),
                acts,
                pi: res.politique.clone(),
                q: res.valeur as f32,
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
        match g.e.gagnant {
            -1 => s.nulles += 1,
            0 => s.victoires_blanc += 1,
            _ => s.victoires_noir += 1,
        }
        for ex in &e.exemples {
            let z = if g.e.gagnant < 0 { 0.0 } else if g.e.gagnant as u8 == ex.equipe { 1.0 } else { -1.0 };
            self.empaqueter(ex, z);
        }
        if self.releves.len() < self.p.releves {
            self.releves.push(Releve { graine: e.graine, actions: e.journal.clone(), resultat: g.resultat() });
        }
    }

    fn empaqueter(&mut self, ex: &Exemple, z: f32) {
        let d = &mut self.donnees;
        d.n += 1;
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
