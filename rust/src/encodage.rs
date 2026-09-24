//! Encodage des positions et des actions pour le réseau, identique à `ia/encodage.py`.

use crate::moteur::*;
use crate::plateau::{plateau, AUCUNE_CASE, N_CASES};

pub const CELL_I: usize = 4;
pub const CELL_F: usize = 3;
pub const UNIT_I: usize = 2;
pub const UNIT_F: usize = 12;
pub const GLOB_I: usize = 2;
pub const GLOB_F: usize = 26;
pub const ACT_F: usize = 7;
pub const N_UNITES: usize = 8;
/// Type d'action « choisir une carte » et type de décision « draft » (glob_i[0]) de l'encodage.
pub const ATT_DRAFT: i64 = 7;

/// Observation du joueur au trait, en entiers (i64) et flottants (f32), à plat.
#[derive(Clone)]
pub struct Observation {
    pub cell_i: [i64; N_CASES * CELL_I],
    pub cell_f: [f32; N_CASES * CELL_F],
    pub unit_i: [i64; N_UNITES * UNIT_I],
    pub unit_f: [f32; N_UNITES * UNIT_F],
    pub glob_i: [i64; GLOB_I],
    pub glob_f: [f32; GLOB_F],
}

#[inline]
fn vue(equipe: u8, c: u8) -> usize {
    if equipe == 0 {
        c as usize
    } else {
        plateau().rotation[c as usize] as usize
    }
}

pub fn encoder_etat(g: &Partie) -> Observation {
    let pl = plateau();
    let p = g.au_trait();
    let me = &g.e.joueurs[p as usize];
    let opp = &g.e.joueurs[1 - p as usize];
    let t = me.equipe;
    let mut o = Observation {
        cell_i: [0; N_CASES * CELL_I],
        cell_f: [0.0; N_CASES * CELL_F],
        unit_i: [0; N_UNITES * UNIT_I],
        unit_f: [0.0; N_UNITES * UNIT_F],
        glob_i: [0; GLOB_I],
        glob_f: [0.0; GLOB_F],
    };
    for &l in &pl.lieux {
        let c = g.e.controle[l as usize];
        o.cell_i[vue(t, l) * CELL_I + 2] = if c < 0 { 1 } else if c as u8 == t { 2 } else { 3 };
    }
    let mut pieces_plateau = [[0i32; N_TYPES]; 2];
    let mut n_unites = [[0i32; N_TYPES]; 2];
    for &pos in g.e.ordre.as_slice() {
        let u = g.e.cases[pos as usize];
        let v = vue(t, pos);
        let cote = if u.proprio == p { 0 } else { 1 };
        o.cell_i[v * CELL_I] = u.genre as i64;
        o.cell_i[v * CELL_I + 1] = 1 + cote as i64;
        o.cell_i[v * CELL_I + 3] = (u.pieces as i64).min(8);
        o.cell_f[v * CELL_F] = u.pieces as f32 / 4.0;
        pieces_plateau[cote][u.genre as usize] += u.pieces as i32;
        n_unites[cote][u.genre as usize] += 1;
    }
    let attente = g.attente().copied();
    if let Some(pd) = attente {
        if pd.pos >= 0 {
            o.cell_f[vue(t, pd.pos as u8) * CELL_F + 1] = 1.0;
        }
        for &pos in pd.positions() {
            o.cell_f[vue(t, pos) * CELL_F + 2] = 1.0;
        }
    }
    let mut k = 0;
    for (cote, j) in [me, opp].into_iter().enumerate() {
        let mut unites = j.unites;
        let n = j.n_unites as usize;
        unites[..n].sort();
        let mine = cote == 0;
        for &u in &unites[..n] {
            o.unit_i[k * UNIT_I] = u as i64;
            o.unit_i[k * UNIT_I + 1] = cote as i64;
            let h = j.main.count(u) as f32;
            let b = j.sac.count(u) as f32;
            let d = j.def_cachee.count(u) as f32;
            let up = j.def_visible.count(u) as f32;
            let f = &mut o.unit_f[k * UNIT_F..(k + 1) * UNIT_F];
            f[0] = if mine { h / 2.0 } else { 0.0 };
            f[1] = if mine { b / 2.0 } else { 0.0 };
            f[2] = if mine { d / 2.0 } else { 0.0 };
            f[3] = (h + b + d) / 3.0;
            f[4] = up / 2.0;
            f[5] = j.reserve[u as usize] as f32 / 3.0;
            f[6] = j.perdues[u as usize] as f32 / 3.0;
            f[7] = pieces_plateau[cote][u as usize] as f32 / 3.0;
            f[8] = n_unites[cote][u as usize] as f32;
            f[9] = NOMBRE[u as usize] as f32 / 5.0;
            f[10] = (h + b + d + up) / 4.0;
            f[11] = if mine && h > 0.0 { 1.0 } else { 0.0 };
            k += 1;
        }
    }
    if g.e.en_draft {
        for u in 1..=16u8 {
            if g.e.dispo & (1 << (u - 1)) != 0 {
                o.unit_i[k * UNIT_I] = u as i64;
                o.unit_i[k * UNIT_I + 1] = 2;
                o.unit_f[k * UNIT_F + 9] = NOMBRE[u as usize] as f32 / 5.0;
                k += 1;
            }
        }
    }
    if let Some(pd) = attente {
        o.glob_i[0] = pd.genre as i64;
        o.glob_i[1] = pd.piece as i64;
    } else if g.e.en_draft {
        o.glob_i[0] = ATT_DRAFT;
    }
    let gf = &mut o.glob_f;
    let un = |b: bool| if b { 1.0f32 } else { 0.0 };
    gf[0] = g.e.marqueurs[t as usize] as f32 / 6.0;
    gf[1] = g.e.marqueurs[1 - t as usize] as f32 / 6.0;
    gf[2] = un(g.e.initiative == p);
    gf[3] = un(g.e.init_bougee);
    gf[4] = un(g.e.premier_manche == p);
    gf[5] = un(g.peut_prendre_initiative(p));
    gf[6] = me.main.len() as f32 / 3.0;
    gf[7] = opp.main.len() as f32 / 3.0;
    gf[8] = me.sac.len() as f32 / 10.0;
    gf[9] = opp.sac.len() as f32 / 10.0;
    gf[10] = (me.def_visible.len() + me.def_cachee.len()) as f32 / 10.0;
    gf[11] = opp.def_cachee.len() as f32 / 5.0;
    gf[12] = opp.def_visible.len() as f32 / 10.0;
    gf[13] = un(me.main.contains(ROYAL));
    gf[14] = un(me.sac.contains(ROYAL));
    gf[15] = un(me.def_cachee.contains(ROYAL) || me.def_visible.contains(ROYAL));
    gf[16] = un(opp.sac.contains(ROYAL) || opp.main.contains(ROYAL) || opp.def_cachee.contains(ROYAL));
    gf[17] = un(opp.def_visible.contains(ROYAL));
    gf[18] = (g.e.manche.min(100)) as f32 / 50.0;
    gf[19] = g.e.manche as f32 / g.e.max_manches as f32;
    gf[20] = un(g.e.courant == p);
    gf[21] = pl.lieux.iter().filter(|&&l| g.e.controle[l as usize] < 0).count() as f32 / 10.0;
    if g.e.en_draft {
        let s = g.e.etape_draft as usize;
        let mut n = 1;
        while s + n < N_CARTES && ORDRE_DRAFT[s + n] == ORDRE_DRAFT[s] {
            n += 1;
        }
        gf[22] = 1.0;
        gf[23] = s as f32 / N_CARTES as f32;
        gf[24] = n as f32 / 2.0;
    }
    gf[25] = un(g.e.premier == p);
    o
}

/// Écrit les 7 entiers de chaque action (type, pièce, unité, extra, 3 cases vues).
pub fn encoder_actions(g: &Partie, legal: &[Action], out: &mut Vec<i64>) {
    let t = g.equipe(g.au_trait());
    for a in legal {
        out.push(a.genre as i64);
        out.push(a.piece as i64);
        out.push(a.unite as i64);
        out.push(a.extra as i64);
        for k in 0..3 {
            out.push(if k < a.nc as usize { vue(t, a.cases[k]) as i64 } else { AUCUNE_CASE as i64 });
        }
    }
}
