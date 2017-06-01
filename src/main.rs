extern crate rand;
extern crate ramp;
extern crate num_traits;
extern crate primal;

use rand::OsRng;
use std::str::FromStr;
use ramp::Int;
use ramp::RandomInt;
use num_traits::{Zero, One};

fn sample(bitsize: usize) -> Int {
    let mut rng = OsRng::new().unwrap();
    rng.gen_uint(bitsize)
}

fn sample_below(upper: &Int) -> Int {
    let mut rng = OsRng::new().unwrap();
    rng.gen_uint_below(upper)
}

fn sample_range(lower: &Int, upper: &Int) -> Int {
    let mut rng = OsRng::new().unwrap();
    rng.gen_int_range(lower, upper)
}

fn sample_prime(bitsize: usize) -> Int {
    // See Practical Considerations section inside the section 11.5 "Prime Number Generation"
    // Applied Cryptography, Bruce Schneier.
    let one = Int::one();
    let two = &one + &one;

    loop {
        let mut candidate = sample(bitsize);
        
        // We flip the LSB to make sure the candidate is odd.
        candidate.set_bit(0, true);

        // To ensure the appropiate size we set the MSB of the candidate.
        candidate.set_bit((bitsize-1) as u32, true);
        
        // If no prime number is found in 500 iterations, restart the loop (re-seed).
        for _ in 0..500 {
            if is_prime(&candidate) {
                return candidate;
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
fn is_prime(candidate: &Int) -> bool
{
    // First, simple trial divide
    for p in primal::Primes::all().take(2048) {
        let p = Int::from(p);
        if *candidate == p {
            return true
        } else {
            let r = candidate % &p;
            if !r.is_zero() {
                continue
            } else {
                return false
            }
        }
    }
    // Second, do a little Fermat test on the candidate
    if !fermat(candidate) {
        return false
    }

    // Finally, do a Miller-Rabin test
    // NIST recommendation is 5 rounds for 512 and 1024 bits. For 1536 bits, the recommendation is 4 rounds.
    if !miller_rabin(candidate, 5) {
        return false
    }
    true
}

fn modpow(base: &Int, exponent: &Int, modulus: &Int) -> Int 
{
    let mut base = base.clone();
    let mut exponent = exponent.clone();
    let mut result = Int::one();

    while !Int::is_zero(&exponent) {
        if !Int::is_even(&exponent) {
            result = (&result * &base) % modulus;
        }
        base = (&base * &base) % modulus;  // waste one of these by having it here but code is simpler (tiny bit)
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
fn miller_rabin(candidate: &Int, limit: usize) -> bool
{
    let (s, d) = rewrite(&(candidate - Int::one()));
    let one = Int::one();
    let two = &one + &one;

    for _ in 0..limit {
        let basis = sample_range(&two, &(candidate-&two));
        let mut y = modpow(&basis, &d, candidate);

        if y == one || y == (candidate - &one) {
            continue;
        } else {
            let mut counter = Int::one();
            while counter < (&s-&one){
                y = modpow(&y, &two, candidate);
                if y == one {
                    return false
                } else if y == candidate - &one {
                    break;
                }
                counter = counter + Int::one();
            }
            return false;
        }
    }
    true
}

// rewrites a number n =  2^s * d
// (i.e., 2^s is the largest power of 2 that divides the candidate).
fn rewrite(n: &Int) -> (Int, Int)
{
     let mut d = n.clone();
     let mut s = Int::zero();
     let one = Int::one();

     while Int::is_even(&d) {
         d = d >> 1_usize;
         s = &s + &one;
     }
     (s,d)
}

fn remove_factor(factor: &Int, mut m: Int) -> Int {
    let mut exponent: usize = 0;
    while (m != 1) && (&m % factor == 0) {
        m /= factor;
        exponent += 1;
    }
    return m
}

#[test]
fn test_remove_factor() {
    assert_eq![remove_factor(&Int::from(2), Int::from(5)), Int::from(5)];
    assert_eq![remove_factor(&Int::from(2), Int::from(6)), Int::from(3)];
    assert_eq![remove_factor(&Int::from(3), Int::from(9)), Int::from(1)];
}

fn prime_factor(n: &Int) -> Vec<Int> {    
    assert!(*n >= Int::from(1));
    
    let mut prime_factors = vec![];
    
    let mut remaining = n.clone();
    let mut factor = Int::from(2);
    
    for q in primal::Primes::all() {
        if remaining == 1 { break }
        
        factor = Int::from(q);
        if &remaining % &factor == 0 {
            prime_factors.push(factor.clone());
            println!("found factor {:?}", factor);
            remaining = remove_factor(&factor, remaining);
        }
    }
    
    println!("Trying large primes");
    loop {
        if remaining == 1 { break }
        
        if is_prime(&remaining) {
            prime_factors.push(remaining.clone());
            println!("found factor {:?}", factor);
            remaining = remove_factor(&factor, remaining);
        } else {
            loop {
                factor += 1;
                if &remaining % &factor == 0 {
                    prime_factors.push(factor.clone());
                    println!("found factor {:?}", factor);
                    remaining = remove_factor(&factor, remaining);
                    break
                }
            }
        }
    }
    
    return prime_factors
}

#[test]
fn test_prime_factor() {
    assert_eq![
        prime_factor(&Int::from(1)),
        vec![]
    ];
    
    assert_eq![
        prime_factor(&Int::from(2)),
        vec![2].into_iter().map(|f| Int::from(f)).collect::<Vec<_>>()
    ];
    
    assert_eq![
        prime_factor(&Int::from(6)),
        vec![2, 3].into_iter().map(|f| Int::from(f)).collect::<Vec<_>>()
    ];
    
    assert_eq![
        prime_factor(&Int::from(24)),
        vec![2, 3].into_iter().map(|f| Int::from(f)).collect::<Vec<_>>()
    ];
    
    assert_eq![
        prime_factor(&Int::from_str("4297130280").unwrap()),
        vec![2, 3, 5, 2281, 5233].into_iter().map(|f| Int::from(f)).collect::<Vec<_>>()
    ];
    
    // assert_eq![
    //     prime_factor(&Int::from_str("18446744073713353129").unwrap()),
    //     vec![3, 5, 2281, 5233].into_iter().map(|f| Int::from(f)).collect::<Vec<_>>()
    // ];
}

fn find_prime(min_size: &Int, n: &Int, m: &Int) -> Int
{
    let mut k = (min_size - 1) / (n * m) + 1;
    loop {
        let q = &k * n * m;
        let p = q + 1;
        
        if is_prime(&p) { 
            return p
        }
        
        k += 1;
    }
}

fn find_generator(p: &Int) -> Int 
{
    let q = p - 1;
    let prime_factors = prime_factor(&q);
    let mut candidate = Int::from(2);
    loop {
        if prime_factors.iter().all(|factor| {
            let exponent = &q / factor;
            modpow(&candidate, &exponent, p) != Int::one()
        }) {
            return candidate
        }
        candidate += 1;
    }
}

fn find_field(min_size: &Int, n: &Int, m: &Int) -> (Int, Int) {
    let p = find_prime(&min_size, &n, &m);
    let g = find_generator(&p);
    
    let order = &p - 1;
    assert!(&order % n == 0);
    assert!(&order % m == 0);
    
    return (p, g)
}

#[test]
fn test_find_field() {
    assert_eq![
        find_field(&Int::from(200), &Int::from(8), &Int::from(9)),
        (Int::from(433), Int::from(5))
    ];
}


/// Finds a prime p and the prime factors of its order such that p has a subgroup of order c
fn new_find_prime<C: Borrow<Int>>(min_bitsize: usize, c: C) -> (Int, Vec<Int>) {
    loop {
        let k1 = sample_prime(min_bitsize);
        
        for k2 in (1..5000).map(|n| Int::from(n)) {
            let p = &k1 * &k2 * c.borrow() + 1;
            
            if is_prime(&p) {
                let mut prime_factors = vec![k1];
                prime_factors.extend(prime_factor(&k2));
                prime_factors.extend(prime_factor(c.borrow()));
                prime_factors.sort();
                prime_factors.dedup();
                return (p, prime_factors)
            }
        }
    }
}

fn new_find_generator(p: &Int, prime_factors: &[Int]) -> Int 
{
    let q = p - 1;
    let mut candidate = Int::from(2);
    loop {
        if prime_factors.iter().all(|factor| {
            let exponent = &q / factor;
            modpow(&candidate, &exponent, p) != Int::one()
        }) {
            return candidate
        }
        candidate += 1;
    }
}

use std::borrow::Borrow;
fn new_find_field<N: Borrow<Int>, M: Borrow<Int>>(min_bitsize: usize, n: N, m: M) -> (Int, Int) {
    let (p, prime_factors) = new_find_prime(min_bitsize, n.borrow() * m.borrow());
    println!("{:?}", prime_factors);
    
    let g = new_find_generator(&p, &prime_factors);
    
    let order = &p - 1;
    assert!(&order % n.borrow() == 0);
    assert!(&order % m.borrow() == 0);
    
    return (p, g)
}



fn main() {
    let min_size = Int::one() << 128;
    // let min_size = Int::one() << 64;
    // let min_size = Int::from(200);
    let n = Int::from(8);
    let m = Int::from(9);
    
    let (p, g) = new_find_field(128, &n, &m);
    println!("Prime is {:?}", p);
    println!("Generator is {:?}", g);
    
    let order = &p - Int::one();
    let exponent_n = &order / &n;
    let exponent_m = &order / &m;
    let omega_n = modpow(&g, &exponent_n, &p);
    let omega_m = modpow(&g, &exponent_m, &p);
    println!("{:?}-th root is {:?}", n, omega_n);
    println!("{:?}-th root is {:?}", m, omega_m);
}
