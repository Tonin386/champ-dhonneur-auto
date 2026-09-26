//! Module Python `champ_rs` : moteur de règles, encodage et auto-jeu en Rust.
//!
//! * `Jeu` : une partie 2 joueurs (tests différentiels contre le moteur Python) ;
//! * `AutoJeu` : parties d'auto-jeu simultanées ; Python n'intervient que pour évaluer les lots
//!   de positions avec le réseau (voir `champ_dhonneur/ia/autojeu.py`).
//!
//! Les tableaux passent sous forme de `bytes` (petit-boutiste), lus avec `numpy.frombuffer`.

mod autojeu;
mod draft;
mod encodage;
mod matchs;
mod moteur;
mod plateau;
mod recherche;
mod rng;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use autojeu::{assembler_lot, AutoJeu as AutoJeuRs, Lot, ParamsAutoJeu};
use matchs::{Match as MatchRs, ParamsMatch};
use moteur::{code_lettre, lettre, Action, Attente, Case, Partie, Pile, N_CARTES, N_TYPES};
use plateau::N_CASES;
use rng::{Pioche, Rapide};
use draft::Moteur;
use recherche::{ParamsRecherche, Requete};

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
    /// `draft=True` : mise en place avancée (`cartes` : les 8 lettres, `choisit` : premier à
    /// choisir), équivalent de `Game("2J", "draft", seed, pool=cartes, draft_first=choisit)`.
    #[new]
    #[pyo3(signature = (graine, unites_=None, premier=None, max_manches=150, draft=false, cartes=None, choisit=None))]
    fn new(graine: u64, unites_: Option<Vec<String>>, premier: Option<u8>, max_manches: u16, draft: bool,
           cartes: Option<String>, choisit: Option<u8>) -> PyResult<Self> {
        if draft {
            let c = match cartes {
                None => None,
                Some(s) => {
                    let codes: Vec<u8> = s.bytes().filter_map(code_lettre).collect();
                    if codes.len() != moteur::N_CARTES {
                        return Err(PyValueError::new_err(format!("cartes invalides : {s}")));
                    }
                    let mut c = [0u8; moteur::N_CARTES];
                    c.copy_from_slice(&codes);
                    Some(c)
                }
            };
            return Ok(Jeu { p: Partie::nouvelle_draft(graine, c, choisit, max_manches) });
        }
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
        d.set_item("en_draft", e.en_draft)?;
        let dispo: String = (1..=16u8).filter(|&u| e.dispo & (1 << (u - 1)) != 0).map(lettre).collect();
        d.set_item("dispo", dispo)?;
        d.set_item("unites", e.joueurs.iter().map(|j| j.unites().iter().map(|&u| lettre(u)).collect::<String>()).collect::<Vec<_>>())?;
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
            let res: Vec<(char, i8)> = j.unites().iter().map(|&u| (lettre(u), j.reserve[u as usize])).collect();
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

    /// Partie reconstruite depuis un état complet : même format que `etat` (les zones de pièces
    /// dans l'ordre), plus le draft (« draft », « cartes », « etape_draft », « choisit »). Permet
    /// de chercher avec le moteur Rust depuis une partie du moteur Python (bot, analyse) ; les
    /// pioches futures sont tirées par un générateur rapide de graine `graine`.
    #[staticmethod]
    #[pyo3(signature = (etat, max_manches=150, graine=0))]
    fn depuis_etat(etat: &Bound<'_, PyDict>, max_manches: u16, graine: u64) -> PyResult<Jeu> {
        let codes = |s: &str| -> PyResult<Vec<u8>> {
            s.bytes().map(|b| code_lettre(b).ok_or_else(|| PyValueError::new_err(format!("pièce inconnue : {}", b as char)))).collect()
        };
        let requis = |k: &str| -> PyResult<Bound<'_, PyAny>> {
            etat.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("état incomplet : {k}")))
        };
        let mut p = Partie::nouvelle(graine, Some([[1, 2, 3, 4], [5, 6, 7, 8]]), Some(0), max_manches);
        p.rng = Pioche::Rapide(Rapide::new(graine));
        let e = &mut p.e;
        let unites: Vec<String> = requis("unites")?.extract()?;
        #[allow(clippy::type_complexity)]
        let joueurs: Vec<(String, String, String, String, Vec<(String, i8)>, Vec<(String, i8)>)> =
            requis("joueurs")?.extract()?;
        if unites.len() != 2 || joueurs.len() != 2 {
            return Err(PyValueError::new_err("deux joueurs attendus"));
        }
        for (i, j) in e.joueurs.iter_mut().enumerate() {
            let mut u = codes(&unites[i])?;
            u.sort();
            j.equipe = i as u8;
            j.unites = [0; 4];
            j.unites[..u.len()].copy_from_slice(&u);
            j.n_unites = u.len() as u8;
            let (sac, main, dv, dc, res, perdues) = &joueurs[i];
            j.sac = Pile::from_slice(&codes(sac)?);
            j.main = Pile::from_slice(&codes(main)?);
            j.def_visible = Pile::from_slice(&codes(dv)?);
            j.def_cachee = Pile::from_slice(&codes(dc)?);
            j.reserve = [0; N_TYPES];
            for (l, n) in res {
                j.reserve[codes(l)?[0] as usize] = *n;
            }
            j.perdues = [0; N_TYPES];
            for (l, n) in perdues {
                j.perdues[codes(l)?[0] as usize] = *n;
            }
        }
        e.cases = [Case::default(); N_CASES];
        e.ordre = Pile::new();
        let plat: Vec<(u8, u8, String, u8)> = requis("plateau")?.extract()?;
        for (pos, proprio, genre, pieces) in plat {
            e.cases[pos as usize] = Case { occupee: true, proprio, genre: codes(&genre)?[0], pieces };
            e.ordre.push(pos);
        }
        e.controle = [-1; N_CASES];
        let controle: Vec<(u8, i8)> = requis("controle")?.extract()?;
        for (l, c) in controle {
            e.controle[l as usize] = c;
        }
        let marqueurs: Vec<i8> = requis("marqueurs")?.extract()?;
        e.marqueurs = [marqueurs[0], marqueurs[1]];
        e.premier = requis("premier")?.extract()?;
        e.initiative = requis("initiative")?.extract()?;
        e.init_bougee = requis("init_bougee")?.extract()?;
        e.premier_manche = requis("premier_manche")?.extract()?;
        e.manche = requis("manche")?.extract()?;
        e.courant = requis("courant")?.extract()?;
        e.gagnant = requis("gagnant")?.extract()?;
        e.fini = requis("fini")?.extract()?;
        let attentes: Vec<(u8, u8, i8, u8, bool, Vec<u8>)> = requis("attentes")?.extract()?;
        e.n_attentes = attentes.len() as u8;
        for (k, (genre, joueur, pos, piece, pioche, positions)) in attentes.into_iter().enumerate() {
            let mut pp = [0u8; 2];
            pp[..positions.len()].copy_from_slice(&positions);
            e.attentes[k] = Attente { genre, joueur, pos, piece, pioche, positions: pp, npos: positions.len() as u8 };
        }
        e.max_manches = max_manches;
        e.graine = graine;
        e.draft = lire(etat, "draft", false)?;
        e.en_draft = requis("en_draft")?.extract()?;
        let cartes: String = lire(etat, "cartes", String::new())?;
        e.cartes = [0; N_CARTES];
        let c = codes(&cartes)?;
        e.cartes[..c.len().min(N_CARTES)].copy_from_slice(&c[..c.len().min(N_CARTES)]);
        let dispo: String = requis("dispo")?.extract()?;
        e.dispo = codes(&dispo)?.iter().fold(0u16, |m, &u| m | 1 << (u - 1));
        e.etape_draft = lire(etat, "etape_draft", 0u8)?;
        e.choisit = lire(etat, "choisit", 0u8)?;
        Ok(Jeu { p })
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

/// Lot de positions à évaluer -> dict de bytes (lu par `rs.lot_numpy`).
fn lot_py<'py>(py: Python<'py>, lot: &Lot) -> PyResult<Bound<'py, PyDict>> {
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
    Ok(d)
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
            p_draft: lire(params, "p_draft", 0.0f64)?,
            simulations_draft: lire(params, "simulations_draft", 128usize)?,
            meilleur_coup: lire(params, "meilleur_coup", false)?,
            draft_exact: lire(params, "draft_exact", 0usize)?,
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
        Ok(Some(lot_py(py, &lot)?))
    }

    /// Partie suivie pour le direct : (graine, premier joueur, armées, décisions, finie, parties
    /// terminées), ou None s'il n'y a plus de partie en cours.
    fn suivie(&mut self) -> Option<(u64, u8, Vec<String>, Vec<ActionPy>, bool, u64, bool)> {
        let fin = self.a.stats.parties;
        self.a.suivie().map(|s| {
            let armees = s
                .unites
                .iter()
                .map(|u| u.iter().filter(|&&c| c != 0).map(|&c| lettre(c)).collect::<String>())
                .collect();
            (s.graine, s.premier, armees, s.actions.iter().map(vers_py).collect(), s.fini, fin, s.draft)
        })
    }

    /// Parties terminées depuis le dernier appel à `resultats`.
    fn terminees(&self) -> u64 {
        self.a.stats.parties
    }

    /// Exemples (bytes), statistiques et relevés (graine, actions, résultat) des parties terminées
    /// depuis le dernier appel ; les parties en cours continuent (auto-jeu continu).
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
        d.set_item("lieux", PyBytes::new(py, octets(&dn.lieux)))?;
        d.set_item("marge", PyBytes::new(py, octets(&dn.marge)))?;
        d.set_item("main_adv", PyBytes::new(py, octets(&dn.main_adv)))?;
        d.set_item("debut", PyBytes::new(py, octets(&dn.debut)))?;
        // statistiques remises à zéro : auto-jeu continu, lu par morceaux (voir `terminees`)
        let s = std::mem::take(&mut self.a.stats);
        let st = PyDict::new(py);
        st.set_item("parties", s.parties)?;
        st.set_item("victoires_blanc", s.victoires_blanc)?;
        st.set_item("victoires_noir", s.victoires_noir)?;
        st.set_item("nulles", s.nulles)?;
        st.set_item("manches", s.manches)?;
        st.set_item("decisions", s.decisions)?;
        st.set_item("recherches", s.recherches)?;
        st.set_item("evaluations", s.evaluations)?;
        st.set_item("parties_draft", s.parties_draft)?;
        st.set_item("victoires_choisit", s.victoires_choisit)?;
        st.set_item("victoires_premier", s.victoires_premier)?;
        d.set_item("stats", st)?;
        let releves: Vec<(u64, Vec<ActionPy>, &str, bool)> = std::mem::take(&mut self.a.releves)
            .into_iter()
            .map(|r| (r.graine, r.actions.iter().map(vers_py).collect(), r.resultat, r.draft))
            .collect();
        d.set_item("releves", releves)?;
        Ok(d)
    }
}

/// Paramètres de recherche d'un agent d'évaluation : meilleur coup, sans bruit.
fn params_agent(d: &Bound<'_, PyDict>) -> PyResult<ParamsRecherche> {
    let def = ParamsRecherche::default();
    Ok(ParamsRecherche {
        simulations: lire(d, "simulations", 128usize)?,
        m: lire(d, "m", 32usize)?,
        c_visit: lire(d, "c_visit", def.c_visit)?,
        c_scale: lire(d, "c_scale", def.c_scale)?,
        c_puct: lire(d, "c_puct", def.c_puct)?,
        fpu: lire(d, "fpu", def.fpu)?,
        parallele: lire(d, "parallele", 1usize)?,
        coup_gagnant: lire(d, "coup_gagnant", true)?,
        cle_publique: lire(d, "cle_publique", true)?,
        draft_exact: lire(d, "draft_exact", 0usize)?,
        bruit: false,
        bruit_coup: false,
        ..def
    })
}

/// Matchs entre agents à recherche neuronale (voir `matchs.rs`).
#[pyclass(module = "champ_rs")]
struct Match {
    m: MatchRs,
}

#[pymethods]
impl Match {
    /// `parties` : (graine, agent du camp 0, agent du camp 1) ; `agents` : réglages de recherche.
    #[new]
    #[pyo3(signature = (parties, agents, draft=false, max_manches=150, simultanees=512, releves=0, fils=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(parties: Vec<(u64, usize, usize)>, agents: Vec<Bound<'_, PyDict>>, draft: bool,
           max_manches: u16, simultanees: usize, releves: usize, fils: usize) -> PyResult<Self> {
        let agents: Vec<ParamsRecherche> = agents.iter().map(params_agent).collect::<PyResult<_>>()?;
        if parties.iter().any(|&(_, a, b)| a >= agents.len() || b >= agents.len()) {
            return Err(PyValueError::new_err("agent inconnu"));
        }
        let p = ParamsMatch { parties, draft, max_manches, simultanees: simultanees.max(1), agents, releves,
                              fils: fils.max(1) };
        Ok(Match { m: MatchRs::new(p) })
    }

    /// Envoie les réponses aux lots précédents (par agent : (logits en bytes f32, a_len,
    /// valeurs en bytes f32), ou None pour un lot vide) et renvoie les lots suivants (un par
    /// agent, None s'il est vide), ou None quand toutes les parties sont terminées.
    #[pyo3(signature = (reponses=None))]
    fn etape<'py>(&mut self, py: Python<'py>, reponses: Option<Vec<Option<(Vec<u8>, usize, Vec<u8>)>>>)
                  -> PyResult<Option<Vec<Option<Bound<'py, PyDict>>>>> {
        let conv: Vec<Option<(Vec<f32>, usize, Vec<f32>)>> = reponses
            .unwrap_or_default()
            .into_iter()
            .map(|r| r.map(|(l, a, v)| (flottants(&l), a, flottants(&v))))
            .collect();
        let mut reps: Vec<Option<(&[f32], usize, &[f32])>> =
            conv.iter().map(|r| r.as_ref().map(|(l, a, v)| (l.as_slice(), *a, v.as_slice()))).collect();
        reps.resize(self.m.evaluations.len(), None);
        let Some(lots) = self.m.etape(&reps) else { return Ok(None) };
        lots.iter().map(|l| if l.b > 0 { lot_py(py, l).map(Some) } else { Ok(None) }).collect::<PyResult<_>>().map(Some)
    }

    /// Parties terminées depuis le dernier appel : (graine, agent du camp 0, agent du camp 1,
    /// score du camp 0 (1, 0, -1), manches, décisions, résultat, décisions jouées ou None).
    #[allow(clippy::type_complexity)]
    fn resultats(&mut self) -> Vec<(u64, usize, usize, i8, u16, u32, &'static str, Option<Vec<ActionPy>>)> {
        std::mem::take(&mut self.m.resultats)
            .into_iter()
            .map(|r| (r.graine, r.agents[0], r.agents[1], r.score0, r.manches, r.decisions, r.resultat,
                      r.actions.map(|a| a.iter().map(vers_py).collect())))
            .collect()
    }

    /// Évaluations demandées par chaque agent.
    fn evaluations(&self) -> Vec<u64> {
        self.m.evaluations.clone()
    }

    /// Parties lancées (terminées ou en cours).
    fn lancees(&self) -> usize {
        self.m.lancees()
    }
}

/// Recherche Gumbel IS-MCTS sur une position (bot de jeu, analyse) : même protocole que
/// `AutoJeu.etape`, avec une seule recherche.
#[pyclass(module = "champ_rs")]
struct RechercheJeu {
    r: Moteur,
    requetes: Vec<Requete>,
    fini: bool,
}

#[pymethods]
impl RechercheJeu {
    /// `agent` : réglages de recherche (simulations, m, parallele, c_scale, coup_gagnant…).
    #[new]
    #[pyo3(signature = (jeu, agent, graine=0))]
    fn new(jeu: &Jeu, agent: &Bound<'_, PyDict>, graine: u64) -> PyResult<Self> {
        let p = params_agent(agent)?;
        let mut legal = Vec::new();
        jeu.p.actions_legales(&mut legal);
        if jeu.p.e.fini || legal.is_empty() {
            return Err(PyValueError::new_err("aucune décision à chercher"));
        }
        let (r, requetes) = Moteur::nouvelle(&jeu.p, p.simulations, p, graine);
        Ok(RechercheJeu { r, requetes, fini: false })
    }

    /// Envoie les réponses au lot précédent et renvoie le lot suivant, ou None quand la
    /// recherche est terminée (voir `resultat`).
    #[pyo3(signature = (logits=None, a_len=0, valeurs=None))]
    fn etape<'py>(&mut self, py: Python<'py>, logits: Option<&[u8]>, a_len: usize,
                  valeurs: Option<&[u8]>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if let (Some(l), Some(v)) = (logits, valeurs) {
            let (lg, va) = (flottants(l), flottants(v));
            let reps: Vec<(&[f32], f32)> = (0..va.len()).map(|i| (&lg[i * a_len..(i + 1) * a_len], va[i])).collect();
            self.requetes = self.r.repondre(&reps);
            self.fini = self.requetes.is_empty();
        }
        if self.fini {
            return Ok(None);
        }
        let lot = assembler_lot(vec![std::mem::take(&mut self.requetes)]);
        Ok(Some(lot_py(py, &lot)?))
    }

    /// Arrêt demandé : réponses au dernier lot, puis résultat sans autre simulation.
    fn interrompre(&mut self, logits: &[u8], a_len: usize, valeurs: &[u8]) {
        if self.fini {
            return;
        }
        let (lg, va) = (flottants(logits), flottants(valeurs));
        let reps: Vec<(&[f32], f32)> = (0..va.len()).map(|i| (&lg[i * a_len..(i + 1) * a_len], va[i])).collect();
        self.r.repondre_et_terminer(&reps);
        self.requetes.clear();
        self.fini = true;
    }

    /// Recherche progressive : passe suivante de `budget` simulations sur les `candidats`
    /// meilleurs coups, l'arbre conservé ; renvoie son premier lot (None : rien à chercher).
    fn prolonger<'py>(&mut self, py: Python<'py>, budget: usize, candidats: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
        if !self.fini {
            return Err(PyRuntimeError::new_err("passe précédente non terminée"));
        }
        self.requetes = self.r.prolonger(budget, candidats);
        self.fini = self.requetes.is_empty();
        if self.fini {
            return Ok(None);
        }
        let lot = assembler_lot(vec![std::mem::take(&mut self.requetes)]);
        Ok(Some(lot_py(py, &lot)?))
    }

    /// Ligne principale après chaque coup de la racine (ordre des actions légales) : suite des
    /// coups les plus visités, avec leurs visites. Pièce 255 : pièce face cachée d'un autre joueur.
    fn lignes(&self, profondeur: usize) -> Vec<Vec<(ActionPy, f64)>> {
        self.r.lignes(profondeur).iter().map(|l| l.iter().map(|(a, n)| (vers_py(a), *n)).collect()).collect()
    }

    /// (action, actions légales, politique améliorée π', visites, Q, valeur, valeur du réseau,
    /// simulations), Q et valeurs du point de vue du joueur au trait.
    #[allow(clippy::type_complexity)]
    fn resultat(&self) -> PyResult<(ActionPy, Vec<ActionPy>, Vec<f32>, Vec<f64>, Vec<f64>, f64, f64, usize)> {
        let r = self.r.resultat().ok_or_else(|| PyRuntimeError::new_err("recherche non terminée"))?;
        Ok((vers_py(&r.action), r.legal.iter().map(vers_py).collect(), r.politique.clone(), r.visites.clone(),
            r.q.clone(), r.valeur, r.v_reseau, r.simulations))
    }
}

#[pymodule]
fn champ_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Jeu>()?;
    m.add_class::<AutoJeu>()?;
    m.add_class::<Match>()?;
    m.add_class::<RechercheJeu>()?;
    m.add("NOMS_TYPES", moteur::NOMS_TYPES.to_vec())?;
    Ok(())
}
