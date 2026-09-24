//! Géométrie du plateau 2 joueurs (37 cases), calculée comme `board.py`.

use std::sync::OnceLock;

pub const N_CASES: usize = 37;
pub const AUCUNE_CASE: u8 = N_CASES as u8; // index « pas de case » de l'encodage

const LONGUEURS: [usize; 7] = [4, 5, 6, 7, 6, 5, 4];
const DIRECTIONS: [(i32, i32); 6] = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)];
const DEPARTS: [[&str; 2]; 2] = [["b1", "e1"], ["c6", "f5"]];
const NEUTRES: [&str; 6] = ["a2", "b4", "c3", "e4", "f2", "g3"];

pub struct Plateau {
    pub noms: Vec<String>,
    pub voisins: Vec<Vec<u8>>,
    pub rayons: Vec<Vec<Vec<u8>>>,      // par direction (éventuellement vide)
    pub lignes2: Vec<Vec<(u8, u8)>>,    // (milieu, bout) des rayons de longueur ≥ 2
    pub dist: Vec<[u8; N_CASES]>,
    pub anneau2: Vec<Vec<u8>>,          // cases à distance exactement 2
    pub dans2: Vec<Vec<u8>>,            // cases à distance 1 ou 2
    pub lieux: Vec<u8>,                 // triés
    pub est_lieu: [bool; N_CASES],
    pub departs: [[u8; 2]; 2],
    pub rotation: [u8; N_CASES],        // rotation de 180° (vue de Noir)
}

impl Plateau {
    fn construire() -> Plateau {
        let maxlen = 7usize;
        let centre = 3i32;
        let mut cases: Vec<(usize, usize)> = Vec::new();
        let mut noms = Vec::new();
        let mut xy2: Vec<(i32, i32)> = Vec::new();
        let mut cube: Vec<(i32, i32, i32)> = Vec::new();
        for (x, &l) in LONGUEURS.iter().enumerate() {
            let offset = maxlen - l;
            for n in 1..=l {
                cases.push((x, n));
                noms.push(format!("{}{}", (b'a' + x as u8) as char, n));
                let y2 = (offset + 2 * (n - 1)) as i32;
                xy2.push((x as i32, y2));
                let q = x as i32;
                let r = (y2 - q - centre % 2).div_euclid(2);
                cube.push((q, r, -q - r));
            }
        }
        let index = |nom: &str| noms.iter().position(|n| n == nom).unwrap() as u8;
        let pos_de = |x: i32, y: i32| xy2.iter().position(|&c| c == (x, y)).map(|i| i as u8);
        let mut voisins = Vec::new();
        let mut rayons = Vec::new();
        let mut lignes2 = Vec::new();
        for &(x, y2) in &xy2 {
            let mut nb = Vec::new();
            let mut rs = Vec::new();
            for &(dx, dy) in &DIRECTIONS {
                if let Some(j) = pos_de(x + dx, y2 + dy) {
                    nb.push(j);
                }
                let mut ray = Vec::new();
                let mut k = 1;
                while let Some(j) = pos_de(x + k * dx, y2 + k * dy) {
                    ray.push(j);
                    k += 1;
                }
                rs.push(ray);
            }
            lignes2.push(rs.iter().filter(|r| r.len() >= 2).map(|r| (r[0], r[1])).collect());
            voisins.push(nb);
            rayons.push(rs);
        }
        let mut dist = vec![[0u8; N_CASES]; N_CASES];
        for i in 0..N_CASES {
            for j in 0..N_CASES {
                let (a, b) = (cube[i], cube[j]);
                dist[i][j] = (a.0 - b.0).abs().max((a.1 - b.1).abs()).max((a.2 - b.2).abs()) as u8;
            }
        }
        let anneau2 = (0..N_CASES).map(|i| (0..N_CASES as u8).filter(|&j| dist[i][j as usize] == 2).collect()).collect();
        let dans2 = (0..N_CASES)
            .map(|i| (0..N_CASES as u8).filter(|&j| { let d = dist[i][j as usize]; d > 0 && d <= 2 }).collect())
            .collect();
        let mut lieux: Vec<u8> = DEPARTS.iter().flatten().chain(NEUTRES.iter()).map(|n| index(n)).collect();
        lieux.sort();
        let mut est_lieu = [false; N_CASES];
        for &l in &lieux {
            est_lieu[l as usize] = true;
        }
        let departs = [[index(DEPARTS[0][0]), index(DEPARTS[0][1])], [index(DEPARTS[1][0]), index(DEPARTS[1][1])]];
        let mut rotation = [0u8; N_CASES];
        for (i, &(x, n)) in cases.iter().enumerate() {
            let xr = 6 - x;
            let nom = format!("{}{}", (b'a' + xr as u8) as char, LONGUEURS[xr] - n + 1);
            rotation[i] = index(&nom);
        }
        Plateau { noms, voisins, rayons, lignes2, dist, anneau2, dans2, lieux, est_lieu, departs, rotation }
    }
}

pub fn plateau() -> &'static Plateau {
    static P: OnceLock<Plateau> = OnceLock::new();
    P.get_or_init(Plateau::construire)
}
