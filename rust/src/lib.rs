//! Module Python `champ_rs` : moteur de règles, encodage et auto-jeu en Rust.
//!
//! * `Jeu` : une partie 2 joueurs (tests différentiels contre le moteur Python) ;
//! * `AutoJeu` : parties d'auto-jeu simultanées ; Python n'intervient que pour évaluer les lots
//!   de positions avec le réseau (voir `champ_dhonneur/ia/autojeu.py`).
//!
//! Les tableaux passent sous forme de `bytes` (petit-boutiste), lus avec `numpy.frombuffer`.

mod autojeu;
mod encodage;
mod moteur;
mod plateau;
mod recherche;
mod rng;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use autojeu::{AutoJeu as AutoJeuRs, ParamsAutoJeu};
use moteur::{code_lettre, lettre, Action, Partie, N_TYPES};

type ActionPy = (u8, u8, u8, u8, Vec<u8>);

fn vers_py(a: &Action) -> ActionPy {
    (a.genre, a.piece, a.unite, a.extra, a.cases().to_vec())
}

fn octets<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY : types numériques simples, lus comme octets (numpy.frombuffer côté Python)
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn flottants(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn unites(u: Option<Vec<String>>) -> PyResult<Option<[[u8; 4]; 2]>> {
    let Some(u) = u else { return Ok(None) };
    if u.len() != 2 {
        return Err(PyValueError::new_err("deux armées attendues"));
    }
    let mut out = [[0u8; 4]; 2];
    for (i, s) in u.iter().enumerate() {
        let codes: Vec<u8> = s.bytes().filter_map(code_lettre).collect();
        if codes.len() != 4 {
            return Err(PyValueError::new_err(format!("armée invalide : {s}")));
        }
        out[i].copy_from_slice(&codes);
    }
    Ok(Some(out))
}

#[pyclass(module = "champ_rs")]
struct Jeu {
    p: Partie,
}

#[pymethods]
impl Jeu {
    #[new]
    #[pyo3(signature = (graine, unites_=None, premier=None, max_manches=150))]
    fn new(graine: u64, unites_: Option<Vec<String>>, premier: Option<u8>, max_manches: u16) -> PyResult<Self> {
        Ok(Jeu { p: Partie::nouvelle(graine, unites(unites_)?, premier, max_manches) })
    }

    fn legales(&self) -> Vec<ActionPy> {
        let mut v = Vec::new();
        self.p.actions_legales(&mut v);
        v.iter().map(vers_py).collect()
    }

    fn jouer(&mut self, genre: u8, piece: u8, unite: u8, extra: u8, cases: Vec<u8>) -> PyResult<()> {
        let mut c = [0u8; 3];
        c[..cases.len()].copy_from_slice(&cases);
        let a = Action { genre, piece, unite, extra, nc: cases.len() as u8, cases: c };
        let mut v = Vec::new();
        self.p.actions_legales(&mut v);
        if !v.contains(&a) {
            return Err(PyValueError::new_err(format!("action illégale : {:?}", a)));
        }
        self.p.jouer(&a).map_err(|e| PyRuntimeError::new_err(e.0))
    }

    /// État complet, pour comparaison avec le moteur Python.
    fn etat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let e = &self.p.e;
        let d = PyDict::new(py);
        d.set_item("manche", e.manche)?;
        d.set_item("courant", e.courant)?;
        d.set_item("au_trait", self.p.au_trait())?;
        d.set_item("fini", e.fini)?;
        d.set_item("gagnant", e.gagnant)?;
        d.set_item("marqueurs", e.marqueurs.to_vec())?;
        d.set_item("premier", e.premier)?;
        d.set_item("initiative", e.initiative)?;
        d.set_item("init_bougee", e.init_bougee)?;
        d.set_item("premier_manche", e.premier_manche)?;
        let pl = plateau::plateau();
        let controle: Vec<(u8, i8)> = pl.lieux.iter().map(|&l| (l, e.controle[l as usize])).collect();
        d.set_item("controle", controle)?;
        let plat: Vec<(u8, u8, char, u8)> = e
            .ordre
            .as_slice()
            .iter()
            .map(|&pos| {
                let c = e.cases[pos as usize];
                (pos, c.proprio, lettre(c.genre), c.pieces)
            })
            .collect();
        d.set_item("plateau", plat)?;
        let zones = |z: &[u8]| z.iter().map(|&c| lettre(c)).collect::<String>();
        let mut joueurs = Vec::new();
        for j in &e.joueurs {
            let res: Vec<(char, i8)> = j.unites.iter().map(|&u| (lettre(u), j.reserve[u as usize])).collect();
            let perdues: Vec<(char, i8)> =
                (1..N_TYPES as u8 - 1).filter(|&u| j.possede(u)).map(|u| (lettre(u), j.perdues[u as usize])).collect();
            joueurs.push((
                zones(j.sac.as_slice()),
                zones(j.main.as_slice()),
                zones(j.def_visible.as_slice()),
                zones(j.def_cachee.as_slice()),
                res,
                perdues,
            ));
        }
        d.set_item("joueurs", joueurs)?;
        let attentes: Vec<(u8, u8, i8, u8, bool, Vec<u8>)> = e.attentes[..e.n_attentes as usize]
            .iter()
            .map(|a| (a.genre, a.joueur, a.pos, a.piece, a.pioche, a.positions().to_vec()))
            .collect();
        d.set_item("attentes", attentes)?;
        Ok(d)
    }

    /// Observation encodée (cell_i, cell_f, unit_i, unit_f, glob_i, glob_f) et actions légales.
    fn encoder(&self) -> (Vec<i64>, Vec<f32>, Vec<i64>, Vec<f32>, Vec<i64>, Vec<f32>, Vec<i64>) {
        let o = encodage::encoder_etat(&self.p);
        let mut legal = Vec::new();
        self.p.actions_legales(&mut legal);
        let mut acts = Vec::new();
        encodage::encoder_actions(&self.p, &legal, &mut acts);
        (o.cell_i.to_vec(), o.cell_f.to_vec(), o.unit_i.to_vec(), o.unit_f.to_vec(), o.glob_i.to_vec(),
         o.glob_f.to_vec(), acts)
    }

    fn determiniser(&self, observateur: u8, graine: u64) -> Jeu {
        let mut r = rng::Rapide::new(graine);
        Jeu { p: self.p.determiniser(observateur, &mut r) }
    }

    #[getter]
    fn fini(&self) -> bool {
        self.p.e.fini
    }

    #[getter]
    fn resultat(&self) -> &'static str {
        self.p.resultat()
    }
}

fn lire<'py, T: for<'a> FromPyObject<'a, 'py>>(d: &Bound<'py, PyDict>, cle: &str, defaut: T) -> PyResult<T> {
    match d.get_item(cle)? {
        Some(v) => v.extract::<T>().map_err(|e| e.into()),
        None => Ok(defaut),
    }
}

#[pyclass(module = "champ_rs")]
struct AutoJeu {
    a: AutoJeuRs,
}

#[pymethods]
impl AutoJeu {
    #[new]
    fn new(params: &Bound<'_, PyDict>, graine: u64) -> PyResult<Self> {
        let p = ParamsAutoJeu {
            parties: lire(params, "parties", 64usize)?,
            simultanees: lire(params, "simultanees", 64usize)?,
            simulations: lire(params, "simulations", 64usize)?,
            simulations_rapides: lire(params, "simulations_rapides", 16usize)?,
            p_complete: lire(params, "p_complete", 0.25f64)?,
            max_manches: lire(params, "max_manches", 80u16)?,
            m: lire(params, "m", 16usize)?,
            parallele: lire(params, "parallele", 1usize)?,
            c_visit: lire(params, "c_visit", 50.0f64)?,
            c_scale: lire(params, "c_scale", 0.1f64)?,
            releves: lire(params, "releves", 0usize)?,
        };
        Ok(AutoJeu { a: AutoJeuRs::new(p, graine) })
    }

    /// Envoie les réponses du lot précédent (logits B×a_len et valeurs B, en bytes f32) et
    /// renvoie le lot suivant (dict de bytes) ou None quand les parties sont terminées.
    #[pyo3(signature = (logits=None, a_len=0, valeurs=None))]
    fn etape<'py>(&mut self, py: Python<'py>, logits: Option<&[u8]>, a_len: usize,
                  valeurs: Option<&[u8]>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let (lg, va);
        let reps = match (logits, valeurs) {
            (Some(l), Some(v)) => {
                lg = flottants(l);
                va = flottants(v);
                Some((&lg[..], a_len, &va[..]))
            }
            _ => None,
        };
        let Some(lot) = self.a.etape(reps) else { return Ok(None) };
        let d = PyDict::new(py);
        d.set_item("b", lot.b)?;
        d.set_item("a_len", lot.a_len)?;
        d.set_item("cell_i", PyBytes::new(py, octets(&lot.cell_i)))?;
        d.set_item("cell_f", PyBytes::new(py, octets(&lot.cell_f)))?;
        d.set_item("unit_i", PyBytes::new(py, octets(&lot.unit_i)))?;
        d.set_item("unit_f", PyBytes::new(py, octets(&lot.unit_f)))?;
        d.set_item("glob_i", PyBytes::new(py, octets(&lot.glob_i)))?;
        d.set_item("glob_f", PyBytes::new(py, octets(&lot.glob_f)))?;
        d.set_item("acts", PyBytes::new(py, octets(&lot.acts)))?;
        d.set_item("n_act", PyBytes::new(py, octets(&lot.n_act)))?;
        Ok(Some(d))
    }

    /// Partie suivie pour le direct : (graine, premier joueur, armées, décisions, finie, parties
    /// terminées), ou None s'il n'y a plus de partie en cours.
    fn suivie(&mut self) -> Option<(u64, u8, Vec<String>, Vec<ActionPy>, bool, u64)> {
        let fin = self.a.stats.parties;
        self.a.suivie().map(|s| {
            let armees = s.unites.iter().map(|u| u.iter().map(|&c| lettre(c)).collect::<String>()).collect();
            (s.graine, s.premier, armees, s.actions.iter().map(vers_py).collect(), s.fini, fin)
        })
    }

    /// Exemples (bytes), statistiques et relevés (graine, actions, résultat) des parties jouées.
    fn resultats<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dn = std::mem::take(&mut self.a.donnees);
        let d = PyDict::new(py);
        d.set_item("n", dn.n)?;
        d.set_item("cell_i", PyBytes::new(py, octets(&dn.cell_i)))?;
        d.set_item("cell_f", PyBytes::new(py, octets(&dn.cell_f)))?;
        d.set_item("unit_i", PyBytes::new(py, octets(&dn.unit_i)))?;
        d.set_item("unit_f", PyBytes::new(py, octets(&dn.unit_f)))?;
        d.set_item("glob_i", PyBytes::new(py, octets(&dn.glob_i)))?;
        d.set_item("glob_f", PyBytes::new(py, octets(&dn.glob_f)))?;
        d.set_item("acts", PyBytes::new(py, octets(&dn.acts)))?;
        d.set_item("pi", PyBytes::new(py, octets(&dn.pi)))?;
        d.set_item("n_act", PyBytes::new(py, octets(&dn.n_act)))?;
        d.set_item("z", PyBytes::new(py, octets(&dn.z)))?;
        d.set_item("q", PyBytes::new(py, octets(&dn.q)))?;
        let s = self.a.stats;
        let st = PyDict::new(py);
        st.set_item("parties", s.parties)?;
        st.set_item("victoires_blanc", s.victoires_blanc)?;
        st.set_item("victoires_noir", s.victoires_noir)?;
        st.set_item("nulles", s.nulles)?;
        st.set_item("manches", s.manches)?;
        st.set_item("decisions", s.decisions)?;
        st.set_item("recherches", s.recherches)?;
        st.set_item("evaluations", s.evaluations)?;
        d.set_item("stats", st)?;
        let releves: Vec<(u64, Vec<ActionPy>, &str)> = std::mem::take(&mut self.a.releves)
            .into_iter()
            .map(|r| (r.graine, r.actions.iter().map(vers_py).collect(), r.resultat))
            .collect();
        d.set_item("releves", releves)?;
        Ok(d)
    }
}

#[pymodule]
fn champ_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Jeu>()?;
    m.add_class::<AutoJeu>()?;
    m.add("NOMS_TYPES", moteur::NOMS_TYPES.to_vec())?;
    Ok(())
}
