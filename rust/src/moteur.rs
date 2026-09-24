//! Moteur de règles 2 joueurs, portage fidèle de `champ_dhonneur/engine.py`.
//!
//! Les ordres d'itération sont ceux du moteur Python (ordre d'insertion des unités sur le
//! plateau, pièces triées avec le Sceau royal en premier, cases de déploiement triées…) :
//! les listes d'actions légales sont identiques, dans le même ordre, et avec le générateur
//! [`PyRandom`] une graine donne exactement la même partie. Des tests différentiels (voir
//! `tests/test_rust.py`) le vérifient sur des milliers de parties.

use crate::plateau::{plateau, N_CASES};
use crate::rng::{Pioche, PyRandom, Rapide};

// ------------------------------------------------------------------ pièces et unités
/// Codes des pièces : 0 = aucune, 1..16 = unités (ordre de `ALL_LETTERS`), 17 = Sceau royal.
pub const LETTRES: &[u8; 16] = b"ABCDEFGHKLMNPRSX";
pub const ROYAL: u8 = 17;
pub const N_TYPES: usize = 18;

pub const U_A: u8 = 1;
pub const U_B: u8 = 2;
pub const U_C: u8 = 3;
pub const U_D: u8 = 4;
pub const U_E: u8 = 5;
pub const U_F: u8 = 6;
pub const U_G: u8 = 7;
pub const U_H: u8 = 8;
pub const U_K: u8 = 9;
pub const U_L: u8 = 10;
pub const U_M: u8 = 11;
pub const U_N: u8 = 12;
pub const U_P: u8 = 13;
pub const U_R: u8 = 14;
pub const U_S: u8 = 15;
pub const U_X: u8 = 16;

/// Nombre de pièces de chaque unité (index = code).
pub const NOMBRE: [i8; N_TYPES] = [0, 4, 5, 4, 5, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 5, 5, 0];

pub fn attaque_normale(u: u8) -> bool {
    u != U_A && u != U_L
}

pub fn max_unites(u: u8) -> usize {
    if u == U_F {
        2
    } else {
        1
    }
}

pub fn code_lettre(l: u8) -> Option<u8> {
    if l == b'*' {
        return Some(ROYAL);
    }
    LETTRES.iter().position(|&c| c == l).map(|i| i as u8 + 1)
}

pub fn lettre(code: u8) -> char {
    if code == ROYAL {
        '*'
    } else {
        LETTRES[code as usize - 1] as char
    }
}

/// Clé de tri des pièces : ordre de `sorted()` en Python (« * » avant les lettres).
fn cle_piece(c: u8) -> u8 {
    if c == ROYAL {
        0
    } else {
        c
    }
}

// ------------------------------------------------------------------ actions
pub const DEPLOY: u8 = 0;
pub const BOLSTER: u8 = 1;
pub const MOVE: u8 = 2;
pub const CONTROL: u8 = 3;
pub const ATTACK: u8 = 4;
pub const TACTIC: u8 = 5;
pub const INITIATIVE: u8 = 6;
pub const RECRUIT: u8 = 7;
pub const PASS: u8 = 8;
pub const SKIP: u8 = 9;
pub const RG_RESERVE: u8 = 10;
pub const RG_UNIT: u8 = 11;
pub const DRAFT: u8 = 12;

pub const NOMS_TYPES: [&str; 13] = [
    "deploy", "bolster", "move", "control", "attack", "tactic", "initiative", "recruit", "pass",
    "skip", "rg_reserve", "rg_unit", "draft",
];

/// Mise en place avancée (livret p.12) : 8 cartes, ordre des choix en décalage par rapport au
/// premier à choisir (A1 B2 A2 B2 A1).
pub const N_CARTES: usize = 8;
pub const ORDRE_DRAFT: [u8; N_CARTES] = [0, 1, 1, 0, 0, 1, 1, 0];

/// Une décision (champs identiques au `Action` Python ; 0 = absent).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Action {
    pub genre: u8,
    pub piece: u8,
    pub unite: u8,
    pub extra: u8,
    pub nc: u8,
    pub cases: [u8; 3],
}

impl Action {
    #[inline]
    pub fn simple(genre: u8, piece: u8) -> Action {
        Action { genre, piece, unite: 0, extra: 0, nc: 0, cases: [0; 3] }
    }
    #[inline]
    fn avec(genre: u8, piece: u8, unite: u8, cases: &[u8], extra: u8) -> Action {
        let mut c = [0u8; 3];
        c[..cases.len()].copy_from_slice(cases);
        Action { genre, piece, unite, extra, nc: cases.len() as u8, cases: c }
    }
    #[inline]
    pub fn cases(&self) -> &[u8] {
        &self.cases[..self.nc as usize]
    }
}

// ------------------------------------------------------------------ petits vecteurs
/// Vecteur de capacité fixe, sans allocation (copie d'une partie = simple memcpy).
#[derive(Clone, Copy)]
pub struct Pile<const CAP: usize> {
    n: u8,
    d: [u8; CAP],
}

impl<const CAP: usize> Pile<CAP> {
    pub const fn new() -> Self {
        Pile { n: 0, d: [0; CAP] }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.n as usize
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.d[..self.n as usize]
    }
    #[inline]
    pub fn push(&mut self, v: u8) {
        assert!((self.n as usize) < CAP, "Pile pleine");
        self.d[self.n as usize] = v;
        self.n += 1;
    }
    #[inline]
    pub fn pop(&mut self) -> Option<u8> {
        if self.n == 0 {
            None
        } else {
            self.n -= 1;
            Some(self.d[self.n as usize])
        }
    }
    #[inline]
    pub fn clear(&mut self) {
        self.n = 0;
    }
    #[inline]
    pub fn count(&self, v: u8) -> usize {
        self.as_slice().iter().filter(|&&x| x == v).count()
    }
    #[inline]
    pub fn contains(&self, v: u8) -> bool {
        self.as_slice().contains(&v)
    }
    /// `list.remove(v)` : retire la première occurrence en conservant l'ordre.
    pub fn remove_first(&mut self, v: u8) -> bool {
        let n = self.n as usize;
        if let Some(i) = self.d[..n].iter().position(|&x| x == v) {
            self.d.copy_within(i + 1..n, i);
            self.n -= 1;
            true
        } else {
            false
        }
    }
    #[inline]
    pub fn swap(&mut self, i: usize, j: usize) {
        self.d.swap(i, j);
    }
    pub fn from_slice(s: &[u8]) -> Self {
        let mut p = Self::new();
        for &v in s {
            p.push(v);
        }
        p
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.d[..self.n as usize]
    }
}

// ------------------------------------------------------------------ état
#[derive(Clone, Copy)]
pub struct Joueur {
    pub equipe: u8,
    pub unites: [u8; 4], // triées ; pendant le draft, les n_unites premières seulement
    pub n_unites: u8,
    pub sac: Pile<24>,
    pub main: Pile<24>,
    pub def_visible: Pile<24>,
    pub def_cachee: Pile<24>,
    pub reserve: [i8; N_TYPES],
    pub perdues: [i8; N_TYPES],
}

impl Joueur {
    pub fn possede(&self, u: u8) -> bool {
        u != 0 && self.unites.contains(&u)
    }
    pub fn unites(&self) -> &[u8] {
        &self.unites[..self.n_unites as usize]
    }
    /// Remplit sac et réserve à partir des unités (mise en place).
    fn remplir_sac(&mut self) {
        self.sac.clear();
        self.reserve = [0; N_TYPES];
        self.perdues = [0; N_TYPES];
        for &u in &self.unites {
            self.sac.push(u);
            self.sac.push(u);
            self.reserve[u as usize] = NOMBRE[u as usize] - 2;
        }
        self.sac.push(ROYAL);
    }
}

#[derive(Clone, Copy, Default)]
pub struct Case {
    pub occupee: bool,
    pub proprio: u8,
    pub genre: u8,
    pub pieces: u8,
}

pub const ATT_AUCUNE: u8 = 0;
pub const ATT_BERSERK: u8 = 1;
pub const ATT_SOLDAT: u8 = 2;
pub const ATT_MERC: u8 = 3;
pub const ATT_FOOTMAN: u8 = 4;
pub const ATT_PRIEST: u8 = 5;
pub const ATT_RG: u8 = 6;

#[derive(Clone, Copy)]
pub struct Attente {
    pub genre: u8,
    pub joueur: u8,
    pub pos: i8,
    pub piece: u8, // 0 = None
    pub pioche: bool,
    pub positions: [u8; 2],
    pub npos: u8,
}

impl Attente {
    fn new(genre: u8, joueur: u8, pos: i8) -> Attente {
        Attente { genre, joueur, pos, piece: 0, pioche: false, positions: [0; 2], npos: 0 }
    }
    pub fn positions(&self) -> &[u8] {
        &self.positions[..self.npos as usize]
    }
}

#[derive(Clone, Copy)]
pub struct Etat {
    pub joueurs: [Joueur; 2],
    pub cases: [Case; N_CASES],
    pub ordre: Pile<16>, // positions occupées dans l'ordre d'insertion (ordre du dict Python)
    pub controle: [i8; N_CASES], // -1 = aucun
    pub marqueurs: [i8; 2],
    pub premier: u8,
    pub initiative: u8,
    pub init_bougee: bool,
    pub premier_manche: u8,
    pub manche: u16,
    pub courant: u8,
    pub attentes: [Attente; 16],
    pub n_attentes: u8,
    pub gagnant: i8, // -1 = aucun
    pub fini: bool,
    pub max_manches: u16,
    pub graine: u64,
    // mise en place avancée
    pub draft: bool,     // partie commencée par un draft
    pub en_draft: bool,  // draft en cours
    pub cartes: [u8; N_CARTES],
    pub dispo: u16,      // bit u-1 : carte u encore disponible
    pub etape_draft: u8,
    pub choisit: u8,     // joueur qui choisit la première carte (A)
}

#[derive(Clone)]
pub struct Partie {
    pub e: Etat,
    pub rng: Pioche,
}

#[derive(Debug)]
pub struct CoupIllegal(pub String);

impl Partie {
    /// Équivalent de `Game("2J", units, seed, first, max_rounds)`.
    pub fn nouvelle(graine: u64, unites: Option<[[u8; 4]; 2]>, premier: Option<u8>, max_manches: u16) -> Partie {
        let mut mise = PyRandom::new(graine as u128);
        let rng = Pioche::Python(Box::new(PyRandom::new(graine as u128 * 7919 + 17)));
        let unites = match unites {
            Some(u) => u,
            None => {
                let mut deck: Vec<u8> = (1..=16).collect();
                mise.shuffle(&mut deck);
                let mut u = [[0u8; 4]; 2];
                for i in 0..2 {
                    let mut g: Vec<u8> = deck[i * 4..(i + 1) * 4].to_vec();
                    g.sort();
                    u[i].copy_from_slice(&g);
                }
                u
            }
        };
        let vide = Joueur {
            equipe: 0,
            unites: [0; 4],
            n_unites: 0,
            sac: Pile::new(),
            main: Pile::new(),
            def_visible: Pile::new(),
            def_cachee: Pile::new(),
            reserve: [0; N_TYPES],
            perdues: [0; N_TYPES],
        };
        let mut joueurs = [vide, vide];
        for (i, j) in joueurs.iter_mut().enumerate() {
            j.equipe = i as u8;
            j.unites = unites[i];
            j.n_unites = 4;
            j.remplir_sac();
        }
        let pl = plateau();
        let mut controle = [-1i8; N_CASES];
        for (t, deps) in pl.departs.iter().enumerate() {
            for &c in deps {
                controle[c as usize] = t as i8;
            }
        }
        let premier = match premier {
            Some(p) => p,
            None => mise.randbelow(2) as u8,
        };
        let e = Etat {
            joueurs,
            cases: [Case::default(); N_CASES],
            ordre: Pile::new(),
            controle,
            marqueurs: [4, 4],
            premier,
            initiative: premier,
            init_bougee: false,
            premier_manche: premier,
            manche: 0,
            courant: premier,
            attentes: [Attente::new(0, 0, -1); 16],
            n_attentes: 0,
            gagnant: -1,
            fini: false,
            max_manches,
            graine,
            draft: false,
            en_draft: false,
            cartes: [0; N_CARTES],
            dispo: 0,
            etape_draft: 0,
            choisit: 0,
        };
        let mut p = Partie { e, rng };
        for (k, &u) in unites[0].iter().chain(unites[1].iter()).enumerate() {
            p.e.cartes[k] = u;
        }
        p.e.cartes.sort();
        p.debut_manche();
        p
    }

    /// Équivalent de `Game("2J", "draft", seed, pool=cartes, draft_first=choisit)`.
    pub fn nouvelle_draft(graine: u64, cartes: Option<[u8; N_CARTES]>, choisit: Option<u8>, max_manches: u16) -> Partie {
        let mut mise = PyRandom::new(graine as u128);
        let mut deck: Vec<u8> = (1..=16).collect();
        mise.shuffle(&mut deck);
        let mut c = match cartes {
            Some(c) => c,
            None => {
                let mut c = [0u8; N_CARTES];
                c.copy_from_slice(&deck[..N_CARTES]);
                c
            }
        };
        c.sort();
        let choisit = match choisit {
            Some(a) => a,
            None => mise.randbelow(2) as u8,
        };
        // armées fictives pour construire l'état, vidées aussitôt (aucun tirage consommé)
        let mut p = Partie::nouvelle(graine, Some([[1, 2, 3, 4], [5, 6, 7, 8]]), Some(1 - choisit), max_manches);
        let rng = Pioche::Python(Box::new(PyRandom::new(graine as u128 * 7919 + 17)));
        p.rng = rng;
        for j in p.e.joueurs.iter_mut() {
            j.unites = [0; 4];
            j.n_unites = 0;
            j.sac.clear();
            j.main.clear();
            j.reserve = [0; N_TYPES];
            j.perdues = [0; N_TYPES];
        }
        p.e.manche = 0;
        p.e.courant = choisit;
        p.e.draft = true;
        p.e.en_draft = true;
        p.e.cartes = c;
        p.e.dispo = c.iter().fold(0u16, |m, &u| m | 1 << (u - 1));
        p.e.etape_draft = 0;
        p.e.choisit = choisit;
        p
    }

    fn choisir_carte(&mut self, u: u8) -> Result<(), CoupIllegal> {
        if u == 0 || u > 16 || self.e.dispo & (1 << (u - 1)) == 0 {
            return Err(CoupIllegal(format!("carte indisponible : {u}")));
        }
        self.e.dispo &= !(1 << (u - 1));
        let j = &mut self.e.joueurs[self.e.courant as usize];
        j.unites[j.n_unites as usize] = u;
        j.n_unites += 1;
        let n = j.n_unites as usize;
        j.unites[..n].sort();
        self.e.etape_draft += 1;
        if (self.e.etape_draft as usize) < N_CARTES {
            self.e.courant = (self.e.choisit + ORDRE_DRAFT[self.e.etape_draft as usize]) % 2;
            return Ok(());
        }
        self.e.en_draft = false;
        for j in self.e.joueurs.iter_mut() {
            j.remplir_sac();
        }
        let b = 1 - self.e.choisit;
        self.e.premier = b;
        self.e.initiative = b;
        self.e.premier_manche = b;
        self.e.courant = b;
        self.debut_manche();
        Ok(())
    }

    /// Copie jetable pour une recherche : pioches tirées par un générateur rapide.
    #[inline]
    pub fn copie_rapide(&self, graine: u64) -> Partie {
        Partie { e: self.e, rng: Pioche::Rapide(Rapide::new(graine)) }
    }

    #[inline]
    pub fn equipe(&self, p: u8) -> u8 {
        self.e.joueurs[p as usize].equipe
    }

    #[inline]
    pub fn au_trait(&self) -> u8 {
        if self.e.n_attentes > 0 {
            self.e.attentes[self.e.n_attentes as usize - 1].joueur
        } else {
            self.e.courant
        }
    }

    pub fn attente(&self) -> Option<&Attente> {
        if self.e.n_attentes > 0 {
            Some(&self.e.attentes[self.e.n_attentes as usize - 1])
        } else {
            None
        }
    }

    #[inline]
    fn empiler(&mut self, a: Attente) {
        assert!((self.e.n_attentes as usize) < self.e.attentes.len(), "pile de décisions pleine");
        self.e.attentes[self.e.n_attentes as usize] = a;
        self.e.n_attentes += 1;
    }

    #[inline]
    fn depiler(&mut self) -> Attente {
        self.e.n_attentes -= 1;
        self.e.attentes[self.e.n_attentes as usize]
    }

    fn unites_de(&self, p: u8, u: u8, out: &mut [u8; 16]) -> usize {
        let mut n = 0;
        for &pos in self.e.ordre.as_slice() {
            let c = &self.e.cases[pos as usize];
            if c.proprio == p && c.genre == u {
                out[n] = pos;
                n += 1;
            }
        }
        n
    }

    #[inline]
    fn libre(&self, c: u8) -> bool {
        !self.e.cases[c as usize].occupee
    }

    // ---------------------------------------------------------------- plateau
    fn poser(&mut self, pos: u8, proprio: u8, genre: u8, pieces: u8) {
        self.e.cases[pos as usize] = Case { occupee: true, proprio, genre, pieces };
        self.e.ordre.push(pos);
    }

    fn enlever(&mut self, pos: u8) {
        self.e.cases[pos as usize] = Case::default();
        self.e.ordre.remove_first(pos);
    }

    fn deplacer(&mut self, de: u8, vers: u8) {
        let c = self.e.cases[de as usize];
        self.enlever(de);
        self.poser(vers, c.proprio, c.genre, c.pieces);
    }

    // ---------------------------------------------------------------- pioche
    fn piocher(&mut self, p: u8) -> u8 {
        let j = &mut self.e.joueurs[p as usize];
        if j.sac.is_empty() {
            let mut sac = Pile::<24>::new();
            for &c in j.def_visible.as_slice().iter().chain(j.def_cachee.as_slice()) {
                sac.push(c);
            }
            j.sac = sac;
            j.def_visible.clear();
            j.def_cachee.clear();
        }
        if j.sac.is_empty() {
            return 0;
        }
        let n = j.sac.len();
        let i = self.rng.randbelow(n);
        let j = &mut self.e.joueurs[p as usize];
        j.sac.swap(i, n - 1);
        j.sac.pop().unwrap()
    }

    fn debut_manche(&mut self) {
        self.e.manche += 1;
        if self.e.manche > self.e.max_manches {
            self.e.fini = true;
            self.e.gagnant = -1;
            return;
        }
        self.e.init_bougee = false;
        self.e.premier_manche = self.e.initiative;
        for k in 0..2u8 {
            let p = (self.e.premier_manche + k) % 2;
            for _ in 0..3 {
                let c = self.piocher(p);
                if c == 0 {
                    break;
                }
                self.e.joueurs[p as usize].main.push(c);
            }
        }
        self.e.courant = self.e.premier_manche;
    }

    fn avancer(&mut self) {
        for k in 1..=2u8 {
            let p = (self.e.courant + k) % 2;
            if !self.e.joueurs[p as usize].main.is_empty() {
                self.e.courant = p;
                return;
            }
        }
        self.debut_manche();
    }

    // ------------------------------------------------------- actions légales
    pub fn actions_legales(&self, out: &mut Vec<Action>) {
        out.clear();
        if self.e.fini {
            return;
        }
        if self.e.en_draft {
            for u in 1..=16u8 {
                if self.e.dispo & (1 << (u - 1)) != 0 {
                    out.push(Action::avec(DRAFT, 0, u, &[], 0));
                }
            }
            return;
        }
        if let Some(pd) = self.attente() {
            let pd = *pd;
            self.actions_attente(&pd, out);
            return;
        }
        let p = self.e.courant;
        let mut pieces: Vec<u8> = self.e.joueurs[p as usize].main.as_slice().to_vec();
        pieces.sort_by_key(|&c| cle_piece(c));
        pieces.dedup();
        for c in pieces {
            self.actions_piece(p, c, out);
        }
    }

    pub fn peut_prendre_initiative(&self, p: u8) -> bool {
        !self.e.en_draft && !self.e.init_bougee && self.e.premier_manche != p && self.equipe(self.e.initiative) != self.equipe(p)
    }

    fn actions_piece(&self, p: u8, piece: u8, out: &mut Vec<Action>) {
        let pl = plateau();
        let j = &self.e.joueurs[p as usize];
        out.push(Action::simple(PASS, piece));
        for &u in j.unites() {
            if j.reserve[u as usize] > 0 {
                out.push(Action::avec(RECRUIT, piece, 0, &[], u));
            }
        }
        if self.peut_prendre_initiative(p) {
            out.push(Action::simple(INITIATIVE, piece));
        }
        let mut pos = [0u8; 16];
        if piece == ROYAL {
            // 1 ou 2 cases (chemin libre, pas forcément en ligne droite) vers un Lieu libre
            // contrôlé par son équipe
            let n = self.unites_de(p, U_G, &mut pos);
            let t = self.equipe(p) as i8;
            for &ps in &pos[..n] {
                let mut dests = 0u64;
                for &mid in &pl.voisins[ps as usize] {
                    if !self.libre(mid) {
                        continue;
                    }
                    dests |= 1 << mid;
                    for &d in &pl.voisins[mid as usize] {
                        if d != ps && self.libre(d) {
                            dests |= 1 << d;
                        }
                    }
                }
                for d in 0..N_CASES as u8 {
                    if dests & (1u64 << d) != 0 && pl.est_lieu[d as usize] && self.e.controle[d as usize] == t {
                        out.push(Action::avec(TACTIC, ROYAL, U_G, &[ps, d], 0));
                    }
                }
            }
            return;
        }
        let u = piece;
        let n = self.unites_de(p, u, &mut pos);
        if n < max_unites(u) {
            let masque = self.cases_deploiement(p, u);
            for c in 0..N_CASES as u8 {
                if masque & (1u64 << c) != 0 {
                    out.push(Action::avec(DEPLOY, piece, u, &[c], 0));
                }
            }
        }
        for &ps in &pos[..n] {
            out.push(Action::avec(BOLSTER, piece, u, &[ps], 0));
            self.manoeuvres(p, ps, piece, false, 0, out);
            self.tactiques(p, ps, piece, out);
        }
        if u == U_F && n == 2 {
            out.push(Action::avec(TACTIC, piece, U_F, &[], 0));
        }
    }

    fn cases_deploiement(&self, p: u8, u: u8) -> u64 {
        let pl = plateau();
        let t = self.equipe(p) as i8;
        let mut m = 0u64;
        for &l in &pl.lieux {
            if self.e.controle[l as usize] == t && self.libre(l) {
                m |= 1 << l;
            }
        }
        if u == U_E {
            for &pos in self.e.ordre.as_slice() {
                let c = &self.e.cases[pos as usize];
                if self.equipe(c.proprio) as i8 == t {
                    for &nb in &pl.voisins[pos as usize] {
                        if self.libre(nb) {
                            m |= 1 << nb;
                        }
                    }
                }
            }
        }
        m
    }

    #[inline]
    fn peut_attaquer(attaquant: &Case, cible: &Case, paye: u8) -> bool {
        // Chevalier : attaquant renforcé, après avoir payé `paye` pièces de sa pile (Berserk)
        !(cible.genre == U_N && attaquant.pieces < 2 + paye)
    }

    #[inline]
    fn ennemi(&self, p: u8, c: u8) -> Option<Case> {
        let t = self.e.cases[c as usize];
        if t.occupee && self.equipe(t.proprio) != self.equipe(p) {
            Some(t)
        } else {
            None
        }
    }

    fn manoeuvres(&self, p: u8, pos: u8, piece: u8, deplacements_seuls: bool, paye: u8, out: &mut Vec<Action>) {
        let pl = plateau();
        let unite = self.e.cases[pos as usize];
        let u = unite.genre;
        for &nb in &pl.voisins[pos as usize] {
            if self.libre(nb) {
                out.push(Action::avec(MOVE, piece, u, &[pos, nb], 0));
            }
        }
        if deplacements_seuls {
            return;
        }
        let t = self.equipe(p);
        if pl.est_lieu[pos as usize] && self.e.controle[pos as usize] != t as i8 && self.e.marqueurs[t as usize] > 0 {
            out.push(Action::avec(CONTROL, piece, u, &[pos], 0));
        }
        if attaque_normale(u) {
            for &nb in &pl.voisins[pos as usize] {
                if let Some(c) = self.ennemi(p, nb) {
                    if Self::peut_attaquer(&unite, &c, paye) {
                        out.push(Action::avec(ATTACK, piece, u, &[pos, nb], 0));
                    }
                }
            }
        }
    }

    fn tactiques(&self, p: u8, pos: u8, piece: u8, out: &mut Vec<Action>) {
        let pl = plateau();
        let unite = self.e.cases[pos as usize];
        let u = unite.genre;
        let ps = pos as usize;
        match u {
            U_C => {
                for &mid in &pl.voisins[ps] {
                    if !self.libre(mid) {
                        continue;
                    }
                    for &tgt in &pl.voisins[mid as usize] {
                        if let Some(c) = self.ennemi(p, tgt) {
                            if Self::peut_attaquer(&unite, &c, 0) {
                                out.push(Action::avec(TACTIC, piece, u, &[pos, mid, tgt], 0));
                            }
                        }
                    }
                }
            }
            U_H => {
                let mut dests = 0u64;
                for &mid in &pl.voisins[ps] {
                    if !self.libre(mid) {
                        continue;
                    }
                    for &d in &pl.voisins[mid as usize] {
                        if self.libre(d) && pl.dist[ps][d as usize] == 2 {
                            dests |= 1 << d;
                        }
                    }
                }
                for d in 0..N_CASES as u8 {
                    if dests & (1u64 << d) != 0 {
                        out.push(Action::avec(TACTIC, piece, u, &[pos, d], 0));
                    }
                }
            }
            U_A => {
                for &tgt in &pl.anneau2[ps] {
                    if let Some(c) = self.ennemi(p, tgt) {
                        if Self::peut_attaquer(&unite, &c, 0) {
                            out.push(Action::avec(TACTIC, piece, u, &[pos, tgt], 0));
                        }
                    }
                }
            }
            U_X => {
                for &(mid, tgt) in &pl.lignes2[ps] {
                    if !self.libre(mid) {
                        continue;
                    }
                    if let Some(c) = self.ennemi(p, tgt) {
                        if Self::peut_attaquer(&unite, &c, 0) {
                            out.push(Action::avec(TACTIC, piece, u, &[pos, tgt], 0));
                        }
                    }
                }
            }
            U_L => {
                for ray in &pl.rayons[ps] {
                    if ray.is_empty() || !self.libre(ray[0]) {
                        continue;
                    }
                    for dist in 1..=2usize {
                        if ray.len() <= dist {
                            break;
                        }
                        if dist == 2 && !self.libre(ray[1]) {
                            break;
                        }
                        if let Some(c) = self.ennemi(p, ray[dist]) {
                            if Self::peut_attaquer(&unite, &c, 0) {
                                out.push(Action::avec(TACTIC, piece, u, &[pos, ray[dist - 1], ray[dist]], 0));
                            }
                        }
                    }
                }
            }
            U_K => {
                let t = self.equipe(p);
                for &ap in &pl.dans2[ps] {
                    let allie = self.e.cases[ap as usize];
                    if !allie.occupee || self.equipe(allie.proprio) != t || !attaque_normale(allie.genre) {
                        continue;
                    }
                    for &tgt in &pl.voisins[ap as usize] {
                        if let Some(c) = self.ennemi(p, tgt) {
                            if Self::peut_attaquer(&allie, &c, 0) {
                                out.push(Action::avec(TACTIC, piece, u, &[pos, ap, tgt], allie.genre));
                            }
                        }
                    }
                }
            }
            U_D => {
                let t = self.equipe(p);
                for &ap in &pl.dans2[ps] {
                    let allie = self.e.cases[ap as usize];
                    if !allie.occupee || self.equipe(allie.proprio) != t {
                        continue;
                    }
                    for &to in &pl.voisins[ap as usize] {
                        if self.libre(to) && pl.dist[ps][to as usize] <= 2 {
                            out.push(Action::avec(TACTIC, piece, u, &[pos, ap, to], allie.genre));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn actions_attente(&self, pd: &Attente, out: &mut Vec<Action>) {
        match pd.genre {
            ATT_RG => {
                out.push(Action::simple(RG_UNIT, 0));
                out.push(Action::simple(RG_RESERVE, 0));
            }
            ATT_PRIEST => self.actions_piece(pd.joueur, pd.piece, out),
            ATT_BERSERK | ATT_MERC => {
                out.push(Action::simple(SKIP, 0));
                // Berserk : la pièce est défaussée de la pile avant la manœuvre
                let paye = if pd.genre == ATT_BERSERK { 1 } else { 0 };
                self.manoeuvres(pd.joueur, pd.pos as u8, 0, false, paye, out);
            }
            ATT_SOLDAT => {
                out.push(Action::simple(SKIP, 0));
                self.manoeuvres(pd.joueur, pd.pos as u8, 0, true, 0, out);
            }
            ATT_FOOTMAN => {
                out.push(Action::simple(SKIP, 0));
                for &pos in pd.positions() {
                    self.manoeuvres(pd.joueur, pos, 0, false, 0, out);
                }
            }
            g => panic!("décision en attente inconnue {g}"),
        }
    }

    fn nb_actions_attente(&self, pd: &Attente) -> usize {
        let mut v = Vec::with_capacity(16);
        self.actions_attente(pd, &mut v);
        v.len()
    }

    // ----------------------------------------------------------- application
    pub fn jouer(&mut self, a: &Action) -> Result<(), CoupIllegal> {
        if self.e.fini {
            return Err(CoupIllegal("La partie est terminée".into()));
        }
        if self.e.en_draft {
            if a.genre != DRAFT {
                return Err(CoupIllegal(format!("choix de carte attendu : {:?}", a)));
            }
            return self.choisir_carte(a.unite);
        }
        let p = self.au_trait();
        if self.e.n_attentes > 0 {
            let pd = self.depiler();
            self.appliquer_attente(&pd, a)?;
        } else {
            self.appliquer_principale(p, a, true)?;
        }
        self.regler();
        Ok(())
    }

    fn regler(&mut self) {
        while !self.e.fini && self.e.n_attentes > 0 {
            if self.attente_valide() {
                return;
            }
            self.depiler();
        }
        if !self.e.fini && self.e.n_attentes == 0 {
            self.avancer();
        }
    }

    /// Vérifie (et met à jour) la décision au sommet de la pile.
    fn attente_valide(&mut self) -> bool {
        let i = self.e.n_attentes as usize - 1;
        let mut pd = self.e.attentes[i];
        let ok = match pd.genre {
            ATT_RG => true,
            ATT_PRIEST => {
                if !pd.pioche {
                    pd.piece = self.piocher(pd.joueur);
                    pd.pioche = true;
                }
                pd.piece != 0
            }
            ATT_FOOTMAN => {
                let mut reste = [0u8; 2];
                let mut n = 0;
                for &pos in pd.positions() {
                    let c = self.e.cases[pos as usize];
                    if c.occupee && c.genre == U_F && c.proprio == pd.joueur {
                        reste[n] = pos;
                        n += 1;
                    }
                }
                pd.positions = reste;
                pd.npos = n as u8;
                self.nb_actions_attente(&pd) > 1
            }
            g => {
                let c = self.e.cases[pd.pos as usize];
                let voulu = match g {
                    ATT_BERSERK => U_B,
                    ATT_SOLDAT => U_S,
                    _ => U_M,
                };
                if !c.occupee || c.genre != voulu || c.proprio != pd.joueur {
                    false
                } else if g == ATT_BERSERK && c.pieces < 2 {
                    false
                } else {
                    self.nb_actions_attente(&pd) > 1
                }
            }
        };
        self.e.attentes[i] = pd;
        ok
    }

    fn appliquer_principale(&mut self, p: u8, a: &Action, de_la_main: bool) -> Result<(), CoupIllegal> {
        let piece = a.piece;
        let pi = p as usize;
        if de_la_main && !self.e.joueurs[pi].main.remove_first(piece) {
            return Err(CoupIllegal(format!("pièce absente de la main : {:?}", a)));
        }
        match a.genre {
            PASS => self.e.joueurs[pi].def_cachee.push(piece),
            INITIATIVE => {
                self.e.joueurs[pi].def_cachee.push(piece);
                self.e.initiative = p;
                self.e.init_bougee = true;
            }
            RECRUIT => {
                let j = &mut self.e.joueurs[pi];
                j.def_cachee.push(piece);
                j.reserve[a.extra as usize] -= 1;
                j.def_visible.push(a.extra);
                if a.extra == U_M {
                    let mut pos = [0u8; 16];
                    let n = self.unites_de(p, U_M, &mut pos);
                    for &ps in &pos[..n] {
                        self.empiler(Attente::new(ATT_MERC, p, ps as i8));
                    }
                }
            }
            DEPLOY => self.poser(a.cases[0], p, a.unite, 1),
            BOLSTER => self.e.cases[a.cases[0] as usize].pieces += 1,
            MOVE | CONTROL | ATTACK | TACTIC => {
                self.e.joueurs[pi].def_visible.push(piece);
                self.manoeuvre(p, a)?;
            }
            _ => return Err(CoupIllegal(format!("{:?}", a))),
        }
        Ok(())
    }

    fn declencher(&mut self, pos: u8, genre: u8) {
        let c = self.e.cases[pos as usize];
        if !c.occupee {
            return;
        }
        if c.genre == U_B {
            self.empiler(Attente::new(ATT_BERSERK, c.proprio, pos as i8));
        } else if c.genre == U_S && genre == ATTACK {
            self.empiler(Attente::new(ATT_SOLDAT, c.proprio, pos as i8));
        } else if c.genre == U_R && (genre == ATTACK || genre == CONTROL) {
            self.empiler(Attente::new(ATT_PRIEST, c.proprio, -1));
        }
    }

    fn manoeuvre(&mut self, p: u8, a: &Action) -> Result<(), CoupIllegal> {
        let c = a.cases;
        match a.genre {
            MOVE => {
                self.deplacer(c[0], c[1]);
                self.declencher(c[1], MOVE);
            }
            CONTROL => {
                self.declencher(c[0], CONTROL);
                self.controler(p, c[0]);
            }
            ATTACK => {
                self.declencher(c[0], ATTACK);
                self.attaquer(c[0], c[1], true);
            }
            TACTIC => match a.unite {
                U_C | U_L => {
                    self.deplacer(c[0], c[1]);
                    self.declencher(c[1], ATTACK);
                    self.attaquer(c[1], c[2], true);
                }
                U_H | U_G => {
                    self.deplacer(c[0], c[1]);
                    self.declencher(c[1], MOVE);
                }
                U_A | U_X => {
                    self.declencher(c[0], ATTACK);
                    self.attaquer(c[0], c[1], false);
                }
                U_K => {
                    self.declencher(c[1], ATTACK);
                    self.attaquer(c[1], c[2], true);
                }
                U_D => {
                    self.deplacer(c[1], c[2]);
                    self.declencher(c[2], MOVE);
                }
                U_F => {
                    let mut pos = [0u8; 16];
                    let n = self.unites_de(p, U_F, &mut pos);
                    let mut pd = Attente::new(ATT_FOOTMAN, p, -1);
                    for (k, &ps) in pos[..n.min(2)].iter().enumerate() {
                        pd.positions[k] = ps;
                    }
                    pd.npos = n.min(2) as u8;
                    self.empiler(pd);
                }
                _ => return Err(CoupIllegal(format!("{:?}", a))),
            },
            _ => return Err(CoupIllegal(format!("{:?}", a))),
        }
        Ok(())
    }

    fn controler(&mut self, p: u8, pos: u8) {
        let t = self.equipe(p);
        let prec = self.e.controle[pos as usize];
        if prec >= 0 {
            self.e.marqueurs[prec as usize] += 1;
        }
        self.e.controle[pos as usize] = t as i8;
        self.e.marqueurs[t as usize] -= 1;
        if self.e.marqueurs[t as usize] == 0 {
            self.e.fini = true;
            self.e.gagnant = t as i8;
            self.e.n_attentes = 0;
        }
    }

    fn retirer_piece(&mut self, pos: u8) {
        let c = &mut self.e.cases[pos as usize];
        c.pieces -= 1;
        let (proprio, genre, reste) = (c.proprio, c.genre, c.pieces);
        self.e.joueurs[proprio as usize].perdues[genre as usize] += 1;
        if reste == 0 {
            self.enlever(pos);
        }
    }

    fn attaquer(&mut self, att: u8, cible: u8, adjacent: bool) {
        let t = self.e.cases[cible as usize];
        if adjacent && t.genre == U_P {
            self.retirer_piece(att);
        }
        if t.genre == U_G && self.e.joueurs[t.proprio as usize].reserve[U_G as usize] > 0 {
            self.empiler(Attente::new(ATT_RG, t.proprio, cible as i8));
        } else {
            self.retirer_piece(cible);
        }
    }

    fn appliquer_attente(&mut self, pd: &Attente, a: &Action) -> Result<(), CoupIllegal> {
        match pd.genre {
            ATT_RG => {
                match a.genre {
                    RG_RESERVE => {
                        let j = &mut self.e.joueurs[pd.joueur as usize];
                        j.reserve[U_G as usize] -= 1;
                        j.perdues[U_G as usize] += 1;
                    }
                    RG_UNIT => self.retirer_piece(pd.pos as u8),
                    _ => return Err(CoupIllegal(format!("{:?}", a))),
                }
                return Ok(());
            }
            ATT_PRIEST => return self.appliquer_principale(pd.joueur, a, false),
            _ => {}
        }
        if a.genre == SKIP {
            return Ok(());
        }
        match pd.genre {
            ATT_BERSERK => {
                self.e.cases[pd.pos as usize].pieces -= 1;
                self.e.joueurs[pd.joueur as usize].def_visible.push(U_B);
                self.manoeuvre(pd.joueur, a)?;
            }
            ATT_SOLDAT => self.deplacer(a.cases[0], a.cases[1]),
            ATT_MERC => self.manoeuvre(pd.joueur, a)?,
            ATT_FOOTMAN => {
                let mut reste = Attente::new(ATT_FOOTMAN, pd.joueur, -1);
                for &x in pd.positions() {
                    if x != a.cases[0] {
                        reste.positions[reste.npos as usize] = x;
                        reste.npos += 1;
                    }
                }
                if reste.npos > 0 {
                    self.empiler(reste);
                }
                self.manoeuvre(pd.joueur, a)?;
            }
            g => return Err(CoupIllegal(format!("décision {g}"))),
        }
        Ok(())
    }

    // ---------------------------------------------------------- informations
    /// Copie où l'information cachée à `observateur` est ré-échantillonnée (version « rapide »
    /// de `Game.determinize` : pioches de la copie tirées par un générateur rapide).
    pub fn determiniser(&self, observateur: u8, rng: &mut Rapide) -> Partie {
        let mut g = self.copie_rapide(rng.next_u64());
        for (idx, j) in g.e.joueurs.iter_mut().enumerate() {
            if idx as u8 == observateur {
                continue;
            }
            let mut pool = [0u8; 72];
            let mut n = 0;
            for &c in j.sac.as_slice().iter().chain(j.main.as_slice()).chain(j.def_cachee.as_slice()) {
                pool[n] = c;
                n += 1;
            }
            rng.shuffle(&mut pool[..n]);
            let (nh, nd) = (j.main.len(), j.def_cachee.len());
            j.main = Pile::from_slice(&pool[..nh]);
            j.def_cachee = Pile::from_slice(&pool[nh..nh + nd]);
            j.sac = Pile::from_slice(&pool[nh + nd..n]);
        }
        g
    }

    pub fn resultat(&self) -> &'static str {
        if !self.e.fini {
            "*"
        } else if self.e.gagnant < 0 {
            "1/2-1/2"
        } else if self.e.gagnant == 0 {
            "1-0"
        } else {
            "0-1"
        }
    }
}
