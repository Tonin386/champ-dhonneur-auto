//! Recherche Gumbel IS-MCTS, portage de `ia/recherche.py` sous forme de machine à états.
//!
//! Le générateur Python devient un objet qui émet des lots de requêtes (positions encodées) et
//! reçoit les réponses du réseau (logits, valeur) : [`Recherche::nouvelle`] renvoie la requête
//! de la racine, puis chaque appel à [`Recherche::repondre`] renvoie les requêtes suivantes ; une
//! liste vide signifie que la recherche est terminée (voir [`Recherche::resultat`]).

use crate::encodage::{encoder_actions, encoder_etat, Observation};
use crate::moteur::{Action, Partie};
use crate::rng::Rapide;

#[derive(Clone, Copy)]
pub struct ParamsRecherche {
    pub simulations: usize,
    pub m: usize,
    pub c_visit: f64,
    pub c_scale: f64,
    pub c_puct: f64,
    pub fpu: f64,
    pub bruit: bool,
    pub parallele: usize,
    pub perte_virtuelle: f64,
}

impl Default for ParamsRecherche {
    fn default() -> Self {
        ParamsRecherche {
            simulations: 64,
            m: 16,
            c_visit: 50.0,
            c_scale: 0.1,
            c_puct: 1.25,
            fpu: 0.25,
            bruit: true,
            parallele: 1,
            perte_virtuelle: 1.0,
        }
    }
}

struct Noeud {
    enfants: Vec<(Action, u32)>,
    logits: Option<Vec<(Action, f32)>>,
    n: f64,
    w0: f64,  // somme des valeurs du point de vue de l'équipe 0
    equipe: u8, // équipe qui a choisi l'action menant ici
}

impl Noeud {
    fn new(equipe: u8) -> Noeud {
        Noeud { enfants: Vec::new(), logits: None, n: 0.0, w0: 0.0, equipe }
    }
    #[inline]
    fn enfant(&self, a: &Action) -> Option<u32> {
        self.enfants.iter().find(|(b, _)| b == a).map(|&(_, i)| i)
    }
}

#[inline]
fn logit(lg: &[(Action, f32)], a: &Action) -> Option<f32> {
    lg.iter().find(|(b, _)| b == a).map(|&(_, l)| l)
}

/// Position à évaluer par le réseau.
pub struct Requete {
    pub obs: Observation,
    pub actions: Vec<i64>, // n_act × 7
    pub n_act: usize,
}

struct SimEnAttente {
    chemin: Vec<u32>,
    feuille: u32,
    legal: Vec<Action>,
    signe: f64,
}

pub struct Resultat {
    pub action: Action,
    pub legal: Vec<Action>,
    pub politique: Vec<f32>, // cible π'
    pub visites: Vec<f64>,
    pub q: Vec<f64>,
    pub valeur: f64, // valeur moyenne à la racine (point de vue du joueur)
    pub v_reseau: f64,
    pub simulations: usize,
}

pub struct Recherche {
    p: ParamsRecherche,
    rng: Rapide,
    jeu: Partie,
    moi: u8,
    equipe: u8,
    signe: f64,
    legal: Vec<Action>,
    noeuds: Vec<Noeud>,
    logits: Vec<f64>,
    v_hat: f64,
    gumbel: Vec<f64>,
    cand: Vec<usize>,
    n_phases: usize,
    n_sims: usize,
    utilisees: usize,
    taches: Vec<usize>,
    pos_tache: usize,
    attente: Vec<SimEnAttente>,
    racine_evaluee: bool,
    tampon: Vec<Action>,
    pub resultat: Option<Resultat>,
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

fn requete(g: &Partie, legal: &[Action]) -> Requete {
    let mut actions = Vec::with_capacity(legal.len() * 7);
    encoder_actions(g, legal, &mut actions);
    Requete { obs: encoder_etat(g), actions, n_act: legal.len() }
}

impl Recherche {
    /// Démarre une recherche ; renvoie la requête de la racine.
    pub fn nouvelle(jeu: &Partie, simulations: usize, p: ParamsRecherche, graine: u64) -> (Recherche, Vec<Requete>) {
        let moi = jeu.au_trait();
        let equipe = jeu.equipe(moi);
        let mut legal = Vec::with_capacity(48);
        jeu.actions_legales(&mut legal);
        let req = requete(jeu, &legal);
        let r = Recherche {
            p,
            rng: Rapide::new(graine),
            jeu: jeu.clone(),
            moi,
            equipe,
            signe: if equipe == 0 { 1.0 } else { -1.0 },
            legal,
            noeuds: Vec::new(),
            logits: Vec::new(),
            v_hat: 0.0,
            gumbel: Vec::new(),
            cand: Vec::new(),
            n_phases: 1,
            n_sims: simulations,
            utilisees: 0,
            taches: Vec::new(),
            pos_tache: 0,
            attente: Vec::new(),
            racine_evaluee: false,
            tampon: Vec::with_capacity(48),
            resultat: None,
        };
        (r, vec![req])
    }

    /// Transmet les réponses (logits des actions légales, valeur) aux requêtes précédentes.
    pub fn repondre(&mut self, reponses: &[(&[f32], f32)]) -> Vec<Requete> {
        if !self.racine_evaluee {
            self.racine_evaluee = true;
            let (lg, v) = reponses[0];
            self.initialiser(lg, v as f64);
        } else {
            let attente = std::mem::take(&mut self.attente);
            for (sim, &(lg, v)) in attente.into_iter().zip(reponses) {
                let noeud = &mut self.noeuds[sim.feuille as usize];
                let l = noeud.logits.get_or_insert_with(Vec::new);
                for (a, &x) in sim.legal.iter().zip(lg) {
                    if logit(l, a).is_none() {
                        l.push((*a, x));
                    }
                }
                self.retropropager(&sim.chemin, sim.signe * v as f64);
            }
        }
        self.avancer()
    }

    fn initialiser(&mut self, lg: &[f32], v_hat: f64) {
        let k = self.legal.len();
        self.logits = lg[..k].iter().map(|&x| x as f64).collect();
        self.v_hat = v_hat;
        let mut racine = Noeud::new(1 - self.equipe);
        racine.logits = Some(self.legal.iter().zip(&self.logits).map(|(a, &l)| (*a, l as f32)).collect());
        racine.n = 1.0;
        racine.w0 = self.signe * v_hat;
        self.noeuds.push(racine);
        for i in 0..k {
            let a = self.legal[i];
            self.noeuds.push(Noeud::new(self.equipe));
            self.noeuds[0].enfants.push((a, i as u32 + 1));
        }
        self.gumbel = if self.p.bruit { (0..k).map(|_| self.rng.gumbel()).collect() } else { vec![0.0; k] };
        if k > 1 && self.n_sims > 0 {
            let m = self.p.m.min(k).min(self.n_sims);
            let mut idx: Vec<usize> = (0..k).collect();
            idx.sort_by(|&a, &b| {
                (self.gumbel[b] + self.logits[b]).partial_cmp(&(self.gumbel[a] + self.logits[a])).unwrap()
            });
            idx.truncate(m);
            self.cand = idx;
            self.n_phases = 1usize.max((m as f64).log2().ceil() as usize);
            self.phase();
        } else {
            self.taches.clear();
            self.pos_tache = 0;
        }
    }

    fn phase(&mut self) {
        let nc = self.cand.len();
        let reste = self.n_sims - self.utilisees;
        let per = if nc <= 2 { (reste + nc - 1) / nc } else { 1usize.max(self.n_sims / (self.n_phases * nc)) };
        self.taches.clear();
        'dehors: for _ in 0..per {
            for &i in &self.cand {
                if self.taches.len() >= reste {
                    break 'dehors;
                }
                self.taches.push(i);
            }
        }
        self.pos_tache = 0;
    }

    fn avancer(&mut self) -> Vec<Requete> {
        let k = self.legal.len();
        if !(k > 1 && self.n_sims > 0) {
            self.terminer();
            return Vec::new();
        }
        loop {
            if self.pos_tache >= self.taches.len() {
                self.utilisees += self.taches.len();
                if self.cand.len() == 1 || self.utilisees >= self.n_sims {
                    self.terminer();
                    return Vec::new();
                }
                let q = self.q_complete();
                let sig = self.sigma(&q);
                let sc: Vec<f64> = (0..k).map(|i| self.gumbel[i] + self.logits[i] + sig[i]).collect();
                self.cand.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap()); // tri stable
                let garder = (self.cand.len() + 1) / 2;
                self.cand.truncate(garder);
                self.phase();
                continue;
            }
            let fin = (self.pos_tache + self.p.parallele.max(1)).min(self.taches.len());
            let vague: Vec<usize> = self.taches[self.pos_tache..fin].to_vec();
            self.pos_tache = fin;
            let mut reqs = Vec::new();
            for a0 in vague {
                if let Some(r) = self.simuler(a0) {
                    reqs.push(r);
                }
            }
            if !reqs.is_empty() {
                return reqs;
            }
        }
    }

    #[inline]
    fn ajouter_vl(&mut self, i: u32) {
        let vl = self.p.perte_virtuelle;
        let n = &mut self.noeuds[i as usize];
        n.n += vl;
        n.w0 -= if n.equipe == 0 { vl } else { -vl };
    }

    fn simuler(&mut self, a0: usize) -> Option<Requete> {
        let vl = self.p.parallele > 1;
        let mut g = self.jeu.determiniser(self.moi, &mut self.rng);
        let mut noeud = a0 as u32 + 1;
        let mut chemin = vec![noeud];
        let a = self.legal[a0];
        g.jouer(&a).expect("action de la racine illégale");
        if vl {
            self.ajouter_vl(noeud);
        }
        let mut legal = std::mem::take(&mut self.tampon);
        loop {
            if g.e.fini {
                let v0 = if g.e.gagnant < 0 { 0.0 } else if g.e.gagnant == 0 { 1.0 } else { -1.0 };
                self.retropropager(&chemin, v0);
                self.tampon = legal;
                return None;
            }
            g.actions_legales(&mut legal);
            let tm = g.equipe(g.au_trait());
            let s = if tm == 0 { 1.0 } else { -1.0 };
            let a = if legal.len() > 1 {
                let complet = match &self.noeuds[noeud as usize].logits {
                    None => false,
                    Some(lg) => legal.iter().all(|a| logit(lg, a).is_some()),
                };
                if !complet {
                    let r = requete(&g, &legal);
                    self.attente.push(SimEnAttente { chemin, feuille: noeud, legal: legal.clone(), signe: s });
                    self.tampon = legal;
                    return Some(r);
                }
                self.puct(noeud, &legal, s)
            } else {
                legal[0]
            };
            let ch = match self.noeuds[noeud as usize].enfant(&a) {
                Some(c) => c,
                None => {
                    let c = self.noeuds.len() as u32;
                    self.noeuds.push(Noeud::new(tm));
                    self.noeuds[noeud as usize].enfants.push((a, c));
                    c
                }
            };
            g.jouer(&a).expect("action illégale dans la recherche");
            noeud = ch;
            chemin.push(noeud);
            if vl {
                self.ajouter_vl(noeud);
            }
        }
    }

    fn retropropager(&mut self, chemin: &[u32], v0: f64) {
        let vl = if self.p.parallele > 1 { self.p.perte_virtuelle } else { 0.0 };
        let r = &mut self.noeuds[0];
        r.n += 1.0;
        r.w0 += v0;
        for &i in chemin {
            let n = &mut self.noeuds[i as usize];
            if vl != 0.0 {
                n.n -= vl;
                n.w0 += if n.equipe == 0 { vl } else { -vl };
            }
            n.n += 1.0;
            n.w0 += v0;
        }
    }

    fn puct(&self, noeud: u32, legal: &[Action], s: f64) -> Action {
        let nd = &self.noeuds[noeud as usize];
        let lg = nd.logits.as_ref().unwrap();
        let vals: Vec<f64> = legal.iter().map(|a| logit(lg, a).unwrap() as f64).collect();
        let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = vals.iter().map(|v| (v - mx).exp()).collect();
        let z: f64 = ex.iter().sum();
        let enfants: Vec<Option<&Noeud>> =
            legal.iter().map(|a| nd.enfant(a).map(|i| &self.noeuds[i as usize])).collect();
        let (mut tot, mut masse) = (0.0, 0.0);
        for (i, ch) in enfants.iter().enumerate() {
            if let Some(c) = ch {
                if c.n > 0.0 {
                    tot += c.n;
                    masse += ex[i];
                }
            }
        }
        let parent_q = if nd.n > 0.0 { s * nd.w0 / nd.n } else { 0.0 };
        let fpu = parent_q - self.p.fpu * (masse / z).sqrt();
        let c = self.p.c_puct * tot.max(1.0).sqrt() / z;
        let (mut meilleur, mut best_u) = (0usize, f64::NEG_INFINITY);
        for (i, ch) in enfants.iter().enumerate() {
            let u = match ch {
                Some(c2) if c2.n > 0.0 => s * c2.w0 / c2.n + c * ex[i] / (1.0 + c2.n),
                _ => fpu + c * ex[i],
            };
            if u > best_u {
                meilleur = i;
                best_u = u;
            }
        }
        legal[meilleur]
    }

    fn q_complete(&self) -> Vec<f64> {
        let k = self.legal.len();
        let pri = softmax(&self.logits);
        let mut q = vec![0.0; k];
        let mut vis = vec![false; k];
        let mut somme_n = 0.0;
        for i in 0..k {
            let ch = &self.noeuds[i + 1];
            if ch.n > 0.0 {
                q[i] = self.signe * ch.w0 / ch.n;
                vis[i] = true;
                somme_n += ch.n;
            }
        }
        let v_mix = if somme_n > 0.0 {
            let pv: f64 = (0..k).filter(|&i| vis[i]).map(|i| pri[i]).sum();
            let pq: f64 = (0..k).filter(|&i| vis[i]).map(|i| pri[i] * q[i]).sum();
            (self.v_hat + somme_n * pq / pv.max(1e-12)) / (1.0 + somme_n)
        } else {
            self.v_hat
        };
        for i in 0..k {
            if !vis[i] {
                q[i] = v_mix;
            }
            q[i] = q[i].clamp(-1.0, 1.0);
        }
        q
    }

    fn sigma(&self, q: &[f64]) -> Vec<f64> {
        let k = self.legal.len();
        let max_n = (0..k).map(|i| self.noeuds[i + 1].n).fold(0.0, f64::max);
        let lo = q.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e = (hi - lo).max(1e-8);
        q.iter().map(|&v| (self.p.c_visit + max_n) * self.p.c_scale * (v - lo) / e).collect()
    }

    fn terminer(&mut self) {
        let k = self.legal.len();
        let q = self.q_complete();
        let sig = self.sigma(&q);
        let z: Vec<f64> = (0..k).map(|i| self.logits[i] + sig[i]).collect();
        let pi = softmax(&z);
        let visites: Vec<f64> = (0..k).map(|i| self.noeuds[i + 1].n).collect();
        let meilleur = if k == 1 {
            0
        } else if self.utilisees == 0 {
            (0..k)
                .max_by(|&a, &b| {
                    (self.gumbel[a] + self.logits[a]).partial_cmp(&(self.gumbel[b] + self.logits[b])).unwrap().then(b.cmp(&a))
                })
                .unwrap()
        } else {
            let mut best = self.cand[0];
            for &i in &self.cand {
                if self.gumbel[i] + self.logits[i] + sig[i] > self.gumbel[best] + self.logits[best] + sig[best] {
                    best = i;
                }
            }
            best
        };
        let r = &self.noeuds[0];
        self.resultat = Some(Resultat {
            action: self.legal[meilleur],
            legal: self.legal.clone(),
            politique: pi.iter().map(|&x| x as f32).collect(),
            visites,
            q,
            valeur: self.signe * r.w0 / r.n,
            v_reseau: self.v_hat,
            simulations: self.utilisees,
        });
    }
}
