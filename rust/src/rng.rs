//! Générateurs aléatoires.
//!
//! * [`PyRandom`] reproduit exactement `random.Random` de CPython (Mersenne Twister MT19937,
//!   initialisation `init_by_array` depuis un entier, `getrandbits`, `_randbelow`, `shuffle`) :
//!   une partie jouée par le moteur Rust avec une graine donne les mêmes pioches que le moteur
//!   Python, ce qui permet de rejouer ses relevés et de comparer les deux moteurs coup par coup.
//! * [`Rapide`] (xoshiro256**) sert aux recherches, où seule la qualité statistique compte.

const N: usize = 624;
const M: usize = 397;

#[derive(Clone)]
pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    /// Équivalent de `random.Random(graine)` pour un entier positif.
    pub fn new(graine: u128) -> Self {
        let mut cle: Vec<u32> = Vec::new();
        let mut n = graine;
        while n > 0 {
            cle.push(n as u32);
            n >>= 32;
        }
        if cle.is_empty() {
            cle.push(0);
        }
        let mut r = PyRandom { mt: [0; N], index: N };
        r.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        let longueur = cle.len();
        let mut k = N.max(longueur);
        while k > 0 {
            let prec = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prec.wrapping_mul(1_664_525)).wrapping_add(cle[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                r.mt[0] = r.mt[N - 1];
                i = 1;
            }
            if j >= longueur {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prec = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prec.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                r.mt[0] = r.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        r.mt[0] = 0x8000_0000;
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prec = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = 1_812_433_253u32.wrapping_mul(prec).wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn genrand_u32(&mut self) -> u32 {
        const MATRICE: u32 = 0x9908_b0df;
        const HAUT: u32 = 0x8000_0000;
        const BAS: u32 = 0x7fff_ffff;
        if self.index >= N {
            for kk in 0..N {
                let y = (self.mt[kk] & HAUT) | (self.mt[(kk + 1) % N] & BAS);
                let mut v = self.mt[(kk + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= MATRICE;
                }
                self.mt[kk] = v;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` pour 1 ≤ k ≤ 32.
    fn getrandbits(&mut self, k: u32) -> u32 {
        self.genrand_u32() >> (32 - k)
    }

    /// `_randbelow_with_getrandbits(n)` : entier uniforme dans [0, n), n ≥ 1.
    pub fn randbelow(&mut self, n: usize) -> usize {
        debug_assert!(n >= 1 && n <= u32::MAX as usize);
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits(k) as usize;
            if r < n {
                return r;
            }
        }
    }

    /// `random.shuffle(x)` (Fisher-Yates depuis la fin).
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i + 1);
            x.swap(i, j);
        }
    }
}

/// xoshiro256** : rapide, pour les déterminisations et les pioches simulées des recherches.
#[derive(Clone, Copy)]
pub struct Rapide {
    s: [u64; 4],
}

impl Rapide {
    pub fn new(graine: u64) -> Self {
        let mut z = graine;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *v = x ^ (x >> 31);
        }
        Rapide { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Entier uniforme dans [0, n) (méthode de Lemire, biais négligeable).
    pub fn randbelow(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Flottant uniforme dans [0, 1).
    pub fn uniforme(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Variable de Gumbel standard (comme `numpy.random.Generator.gumbel`).
    pub fn gumbel(&mut self) -> f64 {
        let u = self.uniforme().max(1e-300);
        -(-u.ln()).ln()
    }

    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i + 1);
            x.swap(i, j);
        }
    }
}

/// Générateur des pioches d'une partie : compatible Python pour les vraies parties,
/// rapide pour les copies jetables des recherches.
#[derive(Clone)]
pub enum Pioche {
    Python(Box<PyRandom>),
    Rapide(Rapide),
}

impl Pioche {
    pub fn randbelow(&mut self, n: usize) -> usize {
        match self {
            Pioche::Python(r) => r.randbelow(n),
            Pioche::Rapide(r) => r.randbelow(n),
        }
    }
}
