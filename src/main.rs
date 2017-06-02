extern crate rand;
extern crate ramp;
extern crate num_traits;
extern crate primal;

use std::borrow::Borrow;
use rand::OsRng;
use ramp::Int;
use ramp::RandomInt;
use num_traits::Zero;

#[cfg(test)]
use std::str::FromStr;

fn sample(bitsize: usize) -> Int {
    let mut rng = OsRng::new().unwrap();
    rng.gen_uint(bitsize)
}

fn sample_below<U: Borrow<Int>>(upper: U) -> Int {
    let mut rng = OsRng::new().unwrap();
    rng.gen_uint_below(upper.borrow())
}

fn sample_range<L: Borrow<Int>, U: Borrow<Int>>(lower: L, upper: U) -> Int {
    let mut rng = OsRng::new().unwrap();
    rng.gen_int_range(lower.borrow(), upper.borrow())
}

fn sample_prime(bitsize: usize) -> Int {
    // See Practical Considerations section inside the section 11.5 "Prime Number Generation"
    // Applied Cryptography, Bruce Schneier.
    let two = Int::from(2);

    loop {
        let mut candidate = sample(bitsize);
        
        // We flip the LSB to make sure the candidate is odd.
        candidate.set_bit(0, true);

        // To ensure the appropiate size we set the MSB of the candidate.
        candidate.set_bit((bitsize-1) as u32, true);
        
        // If no prime number is found in 500 iterations, restart the loop (re-seed).
        for _ in 0..500 {
            if is_prime(&candidate) {
                return candidate
            }
            candidate = candidate + &two;
        }
    }
}

// Runs the following three tests on a given `candidate` to determine
// primality:
//
// 1. Divide the candidate by the smaller prime numbers.
// 2. Run Fermat's Little Theorem against the candidate.
// 3. Run five rounds of the Miller-Rabin test on the candidate.
fn is_prime<C: Borrow<Int>>(candidate: C) -> bool
{
    let candidate = candidate.borrow();
    
    // First, simple trial divide
    // TODO testing against a high number of primes is probably overkill; BoringSSL uses 2048
    for p in primal::Primes::all().take(10000) {
        let p = Int::from(p);
        if p == *candidate {
            return true
        }
        if candidate % &p == 0 {
            return false
        }
    }
    
    // Second, do a little Fermat test on the candidate
    if !fermat(candidate) {
        return false
    }

    // Finally, do a Miller-Rabin test
    // NIST recommendation is 5 rounds for 512 and 1024 bits. 
    // For 1536 bits, the recommendation is 4 rounds.
    // TODO how many rounds should we do?
    if !miller_rabin(candidate, 20) {
        return false
    }
    
    true
}

fn modpow<B: Borrow<Int>, E: Borrow<Int>, M: Borrow<Int>>(base: B, exponent: E, modulus: M) -> Int 
{
    let mut base = base.borrow().clone();
    let mut exponent = exponent.borrow().clone();
    
    let mut result = Int::one();
    while !Int::is_zero(&exponent) {
        if !Int::is_even(&exponent) {
            result = (&result * &base) % modulus.borrow();
        }
        base = (&base * &base) % modulus.borrow();  // waste one of these by having it here but code is simpler (tiny bit)
        exponent = exponent >> 1;
    }
    result
}

fn fermat(candidate: &Int) -> bool
{
    // Perform Fermat's little theorem
    // This might be perform more than once. Handbook of Applied Cryptography [Algorithm 4.9 p136]
    let random = sample_below(candidate);
    let result = modpow(&random, &(candidate - Int::one()), candidate);
    result == Int::one()
}

// Iterations recommended for which  p < (1/2)^{80}
//  500 bits => 6 iterations
// 1000 bits => 3 iterations
// 2000 bits => 2 iterations
fn miller_rabin(candidate: &Int, iterations: usize) -> bool
{
    // constants for optimisations
    let one = Int::from(1);
    let two = Int::from(2);
    let three = Int::from(3);
    let minusone = candidate - &one;
    
    // corner cases that otherwise makes sampling fail below
    if candidate ==   &one { return false }
    if candidate ==   &two { return true }
    if candidate == &three { return true }
    
    let (s, d) = rewrite(candidate - &one);
    
    (0..iterations).all(|_| {
        let basis = sample_range(&two, candidate - &one);
        
        let mut y = modpow(&basis, &d, candidate);
        if y == one {
            return true
        }
        
        let mut counter = Int::zero();
        while counter <= (&s - &one) {
            if y == minusone {
                return true
            }
            if y == one {
                return false
            }
            y = modpow(&y, &two, candidate);
            counter = counter + &one;
        }
        
        return false
    })
}

#[test]
fn test_miller_rabin() {
    assert!(miller_rabin(&Int::from(2), 10));
    assert!(miller_rabin(&Int::from(3), 10));
    assert!(miller_rabin(&Int::from(5), 10));
    assert!(miller_rabin(&Int::from(7), 10));
    assert!(miller_rabin(&Int::from(11), 10));
    assert!(miller_rabin(&Int::from(433), 10));
    
    assert!( ! miller_rabin(&Int::from(1), 10));
    assert!( ! miller_rabin(&Int::from(4), 10));
    assert!( ! miller_rabin(&Int::from(9), 10));
    assert!( ! miller_rabin(&Int::from(21), 10));
    assert!( ! miller_rabin(&Int::from(256), 10));
}

// rewrites a number n =  2^s * d
// (i.e., 2^s is the largest power of 2 that divides the candidate).
fn rewrite<N: Borrow<Int>>(n: N) -> (Int, Int)
{
     let mut d = n.borrow().clone();
     let mut s = Int::zero();
     let one = Int::one();

     while Int::is_even(&d) {
         d = d >> 1_usize;
         s = &s + &one;
     }
     
     (s, d)
}

fn remove_factor(factor: &Int, mut m: Int) -> Int 
{
    while (m != 1) && (&m % factor == 0) {
        m /= factor;
    }
    return m
}

#[test]
fn test_remove_factor() {
    assert_eq![remove_factor(&Int::from(2), Int::from(5)), Int::from(5)];
    assert_eq![remove_factor(&Int::from(2), Int::from(6)), Int::from(3)];
    assert_eq![remove_factor(&Int::from(3), Int::from(9)), Int::from(1)];
}

fn prime_factor<N: Borrow<Int>>(n: N) -> Vec<Int> 
{
    assert!(*n.borrow() >= Int::from(1));
    
    let mut prime_factors = vec![];
    let mut remaining = n.borrow().clone();
    
    for q in primal::Primes::all() {
        if remaining == 1 { break }
        
        let factor = Int::from(q);
        if &remaining % &factor == 0 {
            remaining = remove_factor(&factor, remaining);
            prime_factors.push(factor);
        }
    }
    
    // TODO
    // factoring further than what's done above is horribly slow,
    // so assume for now that we never have to use it, ie that inputs
    // are small enough to only have prime factors found above
    assert!(remaining == 1);
    
    prime_factors
}

#[test]
fn test_prime_factor() {
    assert_eq![
        prime_factor(Int::from(1)),
        vec![].into_iter().map(|f: usize| Int::from(f)).collect::<Vec<Int>>()
    ];
    
    assert_eq![
        prime_factor(Int::from(2)),
        vec![2].into_iter().map(|f| Int::from(f)).collect::<Vec<_>>()
    ];
    
    assert_eq![
        prime_factor(Int::from(6)),
        vec![2, 3].into_iter().map(|f| Int::from(f)).collect::<Vec<_>>()
    ];
    
    assert_eq![
        prime_factor(Int::from(24)),
        vec![2, 3].into_iter().map(|f| Int::from(f)).collect::<Vec<_>>()
    ];
    
    assert_eq![
        prime_factor(Int::from_str("4297130280").unwrap()),
        vec![2, 3, 5, 2281, 5233].into_iter().map(|f| Int::from(f)).collect::<Vec<_>>()
    ];
}

// TODO
// TODO returned prime is too large
// TODO

const SLACK_BOUND: usize = 128; // must be >= 2

/// Finds a prime p and the prime factors of its order such that p has a subgroup of order d
fn find_prime<D: Borrow<Int>>(bitsize: usize, d: D) -> (Int, Vec<Int>) 
{
    loop {
        let k1 = sample_prime(bitsize);
        
        // allowing this k2 factor seems to save some sampling (one iteration of loop enough seems)
        for k2 in (1..SLACK_BOUND).map(|n| Int::from(n)) {
            let candidate = &k1 * &k2 * d.borrow() + 1;
            
            if is_prime(&candidate) {     
                let mut prime_factors = vec![k1];
                prime_factors.extend(prime_factor(k2));
                prime_factors.extend(prime_factor(d.borrow()));
                prime_factors.sort();
                prime_factors.dedup();
                return (candidate, prime_factors)
            }
        }
    }
}

/// Finds a generator of Z_p given the prime factors of p-1
fn find_generator<P: Borrow<Int>>(p: P, prime_factors: &[Int]) -> Int 
{
    let q = p.borrow() - 1;
    let mut candidate = Int::from(2);
    loop {
        if prime_factors.iter().all(|factor| {
            let exponent = &q / factor;
            modpow(&candidate, &exponent, p.borrow()) != Int::one()
        }) {
            return candidate
        }
        candidate += 1;
    }
}

fn find_field<D: Borrow<Int>>(bitsize: usize, order_divisor: D) -> (Int, Int) 
{
    let (p, prime_factors) = find_prime(bitsize, order_divisor);
    let g = find_generator(&p, &prime_factors);
    (p, g)
}

#[test]
fn test_find_field() {
    assert_eq![
        find_field(2, Int::from(2*2*2 * 3*3)),
        (Int::from(433), Int::from(5))
    ];
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parameters {
    prime: Int,
    generator: Int,
    omega_n: Int,
    omega_m: Int,
}

fn bit_length(x: usize) -> usize {
    f64::ceil(f64::log(x as f64, 2_f64)) as usize
}

fn find_parameters(desired_bitsize: usize, n: usize, m: usize) -> Parameters
{
    let order_divisor = n * m;
    let recalibrated_bitsize = desired_bitsize - bit_length(order_divisor) + 1;
    
    let (prime, generator) = find_field(recalibrated_bitsize, Int::from(order_divisor));
    assert![prime.bit_length() as usize >= desired_bitsize];
    assert![prime.bit_length() as usize <= desired_bitsize + bit_length(SLACK_BOUND) + 1]; // TODO verify +1 rounding
    
    let order = &prime - 1;
    let exponent_n = &order / Int::from(n);
    let exponent_m = &order / Int::from(m);
    let omega_n = modpow(&generator, &exponent_n, &prime);
    let omega_m = modpow(&generator, &exponent_m, &prime);
    
    assert![ is_prime(&prime) ];
    assert!(&order % Int::from(n) == 0);
    assert!(&order % Int::from(m) == 0);
    assert![ modpow(&omega_n, Int::from(n), &prime) == Int::one() ];
    assert![ modpow(&omega_m, Int::from(m), &prime) == Int::one() ];
    for e in 1..n { assert![ modpow(&omega_n, Int::from(e), &prime) != Int::one() ]; }
    for e in 1..m { assert![ modpow(&omega_m, Int::from(e), &prime) != Int::one() ]; }
    
    return Parameters {
        prime: prime,
        generator: generator,
        omega_n: omega_n,
        omega_m: omega_m,
    }
}

#[test]
fn test_find_parameters() {
    assert_eq![
        find_parameters(2, 8, 9),
        Parameters { 
            prime: Int::from(433), 
            generator: Int::from(5), 
            omega_n: Int::from(354),
            omega_m: Int::from(150),
        }
    ];
}

fn main() {
    let params = find_parameters(128, 2*2*2*2*2*2*2*2, 9*9*9);
    println!("Parameters are {:?}", params);
    assert!(is_prime(&params.prime));
}
