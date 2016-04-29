/*-
 * Copyright (c) 2016 Taylor R. Campbell
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * weakprng: A pseudorandom number generator with 256-bit key and
 * 112-byte state made by a trivial application of the ChaCha8 stream
 * cipher[1].  See below for a full description of the PRNG.
 *
 * `Weak' because it does not provide the property variously known as
 * backtracking resistance, forward secrecy, or key erasure -- knowing
 * the current state enables an adversary to predict past outputs.
 * This renders it unfit for generating crypto keys, but has no
 * consequences for non-adversarial Monte Carlo simulation.
 *
 * Like any unbroken (as of April 2016) stream cipher, ChaCha8 will
 * pass any statistical test you throw at it, e.g. anything in the
 * dieharder suite -- if you knew a test that could distinguish it
 * from uniform random, your paper would be gladly accepted at a
 * prestigious crypto conference.  In contrast, for example, there is
 * a well-known trivial test with high statistical power to
 * distinguish a Mersenne twister from uniform random given 624
 * consecutive outputs.
 *
 * We use the ChaCha family because among cryptographic stream
 * ciphers, it is uniformly fast on all CPUs with tiny cache
 * footprint, it is quick and easy to implement, and it is generally
 * safe to have floating around even if you copy & paste it and use it
 * for crypto purposes.  This is in contrast to, e.g., AES, which has
 * a lower security margin than ChaCha and is practically guaranteed
 * to be much slower and vulnerable to cache-timing attacks if
 * implemented in software.
 *
 * We use a local implementation of ChaCha8 because it is easier to fit
 * that in a single page of code than it is to pull in and use an
 * external library, most of which are designed for encrypting
 * messages, not for evaluating the ChaCha8 PRF directly as we want.
 *
 * [1] Daniel J. Bernstein, `ChaCha, a variant of Salsa20,' Workshop
 * Record of SASC 2008: The State of the Art of Stream Ciphers, 2008.
 * <https://cr.yp.to/papers.html#chacha>
 */

#define	_POSIX_C_SOURCE	200809L
#define	__STDC_CONSTANT_MACROS

#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "weakprng.h"

/* Miscellaneous utilities */

#define	arraycount(A)		(sizeof(A)/sizeof(*(A)))

#define	attr_unused		__attribute__((__unused__))

#define	CTASSERT(X)		CTASSERT0(X, ctassert, __LINE__)
#define	CTASSERT0(X, Y, Z)	CTASSERT1(X, Y, Z)
#define	CTASSERT1(X, Y, Z)	typedef char Y ## Z[(X) ? 1 : -1] attr_unused

#define	roundup2(X, N)		((((X) - 1) | ((N) - 1)) + 1)

static inline uint32_t
le32dec(const void *buf)
{
	const uint8_t *p = (const uint8_t *)buf;

	return
	    ((uint32_t)p[0] << 0) |
	    ((uint32_t)p[1] << 8) |
	    ((uint32_t)p[2] << 16) |
	    ((uint32_t)p[3] << 24);
}

static inline void
le32enc(void *buf, uint32_t v)
{
	uint8_t *p = (uint8_t *)buf;

	*p++ = v; v >>= 8;
	*p++ = v; v >>= 8;
	*p++ = v; v >>= 8;
	*p++ = v;
}

static inline uint32_t
lemaxdec(const void *buf)
{
	const uint8_t *p = (const uint8_t *)buf;
	uintmax_t vh = 0;
	unsigned i;

	for (i = 0; i < sizeof(vh); i++)
		vh |= (uintmax_t)*p++ << (8*i);

	return vh;
}

/* ChaCha8 core */

#define	crypto_core_OUTPUTBYTES	64
#define	crypto_core_INPUTBYTES	16
#define	crypto_core_KEYBYTES	32
#define	crypto_core_CONSTBYTES	16

#define	crypto_core_ROUNDS	8

static uint32_t
rotate(uint32_t u, unsigned c)
{

	return (u << c) | (u >> (32 - c));
}

#define	QUARTERROUND(a, b, c, d) do {					      \
	(a) += (b); (d) ^= (a); (d) = rotate((d), 16);			      \
	(c) += (d); (b) ^= (c); (b) = rotate((b), 12);			      \
	(a) += (b); (d) ^= (a); (d) = rotate((d),  8);			      \
	(c) += (d); (b) ^= (c); (b) = rotate((b),  7);			      \
} while (/*CONSTCOND*/0)

const uint8_t crypto_core_constant32[16] = {
	'e','x','p','a','n','d',' ','3','2','-','b','y','t','e',' ','k',
};

static void
crypto_core(uint8_t *out, const uint8_t *in, const uint8_t *k,
    const uint8_t *c)
{
	uint32_t x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;
	uint32_t j0,j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15;
	int i;

	j0 = x0 = le32dec(c + 0);
	j1 = x1 = le32dec(c + 4);
	j2 = x2 = le32dec(c + 8);
	j3 = x3 = le32dec(c + 12);
	j4 = x4 = le32dec(k + 0);
	j5 = x5 = le32dec(k + 4);
	j6 = x6 = le32dec(k + 8);
	j7 = x7 = le32dec(k + 12);
	j8 = x8 = le32dec(k + 16);
	j9 = x9 = le32dec(k + 20);
	j10 = x10 = le32dec(k + 24);
	j11 = x11 = le32dec(k + 28);
	j12 = x12 = le32dec(in + 0);
	j13 = x13 = le32dec(in + 4);
	j14 = x14 = le32dec(in + 8);
	j15 = x15 = le32dec(in + 12);

	for (i = crypto_core_ROUNDS; i > 0; i -= 2) {
		QUARTERROUND( x0, x4, x8,x12);
		QUARTERROUND( x1, x5, x9,x13);
		QUARTERROUND( x2, x6,x10,x14);
		QUARTERROUND( x3, x7,x11,x15);
		QUARTERROUND( x0, x5,x10,x15);
		QUARTERROUND( x1, x6,x11,x12);
		QUARTERROUND( x2, x7, x8,x13);
		QUARTERROUND( x3, x4, x9,x14);
	}

	le32enc(out + 0, x0 + j0);
	le32enc(out + 4, x1 + j1);
	le32enc(out + 8, x2 + j2);
	le32enc(out + 12, x3 + j3);
	le32enc(out + 16, x4 + j4);
	le32enc(out + 20, x5 + j5);
	le32enc(out + 24, x6 + j6);
	le32enc(out + 28, x7 + j7);
	le32enc(out + 32, x8 + j8);
	le32enc(out + 36, x9 + j9);
	le32enc(out + 40, x10 + j10);
	le32enc(out + 44, x11 + j11);
	le32enc(out + 48, x12 + j12);
	le32enc(out + 52, x13 + j13);
	le32enc(out + 56, x14 + j14);
	le32enc(out + 60, x15 + j15);
}

/* ChaCha self-test */

/*
 * ChaCha20 test vector from RFC 7539, Appendix A.1, p. 29.  ChaCha8
 * and ChaCha12 test vectors derived by rerunning the same code with
 * the number of rounds changed.  Output blocks match test vectors in
 * J. Strombergson, `Test Vectors for the Stream Cipher ChaCha',
 * Internet-Draft, December 2013,
 * <https://tools.ietf.org/html/draft-strombergson-chacha-test-vectors-01>.
 */
static const uint8_t crypto_core_selftest_vector[64] = {
#if crypto_core_ROUNDS == 8
	0x3e,0x00,0xef,0x2f,0x89,0x5f,0x40,0xd6,
	0x7f,0x5b,0xb8,0xe8,0x1f,0x09,0xa5,0xa1,
	0x2c,0x84,0x0e,0xc3,0xce,0x9a,0x7f,0x3b,
	0x18,0x1b,0xe1,0x88,0xef,0x71,0x1a,0x1e,
	0x98,0x4c,0xe1,0x72,0xb9,0x21,0x6f,0x41,
	0x9f,0x44,0x53,0x67,0x45,0x6d,0x56,0x19,
	0x31,0x4a,0x42,0xa3,0xda,0x86,0xb0,0x01,
	0x38,0x7b,0xfd,0xb8,0x0e,0x0c,0xfe,0x42,
#elif crypto_core_ROUNDS == 12
	0x9b,0xf4,0x9a,0x6a,0x07,0x55,0xf9,0x53,
	0x81,0x1f,0xce,0x12,0x5f,0x26,0x83,0xd5,
	0x04,0x29,0xc3,0xbb,0x49,0xe0,0x74,0x14,
	0x7e,0x00,0x89,0xa5,0x2e,0xae,0x15,0x5f,
	0x05,0x64,0xf8,0x79,0xd2,0x7a,0xe3,0xc0,
	0x2c,0xe8,0x28,0x34,0xac,0xfa,0x8c,0x79,
	0x3a,0x62,0x9f,0x2c,0xa0,0xde,0x69,0x19,
	0x61,0x0b,0xe8,0x2f,0x41,0x13,0x26,0xbe,
#elif crypto_core_ROUNDS == 20
	0x76,0xb8,0xe0,0xad,0xa0,0xf1,0x3d,0x90,
	0x40,0x5d,0x6a,0xe5,0x53,0x86,0xbd,0x28,
	0xbd,0xd2,0x19,0xb8,0xa0,0x8d,0xed,0x1a,
	0xa8,0x36,0xef,0xcc,0x8b,0x77,0x0d,0xc7,
	0xda,0x41,0x59,0x7c,0x51,0x57,0x48,0x8d,
	0x77,0x24,0xe0,0x3f,0xb8,0xd8,0x4a,0x37,
	0x6a,0x43,0xb8,0xf4,0x15,0x18,0xa1,0x1c,
	0xc3,0x87,0xb6,0x69,0xb2,0xee,0x65,0x86,
#else
#  error crypto_core_ROUNDS must be 8, 12, or 20.
#endif
};

static int
crypto_core_selftest(void)
{
	const uint8_t nonce[crypto_core_INPUTBYTES] = {0};
	const uint8_t key[crypto_core_KEYBYTES] = {0};
	uint8_t block[64];
	unsigned i;

	crypto_core(block, nonce, key, crypto_core_constant32);
	for (i = 0; i < 64; i++) {
		if (block[i] != crypto_core_selftest_vector[i])
			return -1;
	}

	return 0;
}

/* PRNG */

/*
 * PRNG output is ChaCha8_seed(0) || ChaCha8_seed(1) || ..., buffered
 * one 64-byte block at a time and yielded one 32-bit word at a time.
 * Unused parts of partial words are discarded by crypto_weakprng_buf.
 *
 * Under the conjecture that ChaCha8 is a PRF, the distribution on
 * PRNG outputs induced by a uniform distribution on seeds is
 * computationally indistinguishable from uniform random.  These PRNG
 * outputs are designed to be fit to serve as one-time pads against an
 * adversary, so they are surely fit for Monte Carlo simulation.
 *
 * This PRNG does not provide what is variously called backtracking
 * resistance, forward secrecy, or key erasure -- don't use it to
 * generate ephemeral key material that you expect to want to erase
 * after a session.  Hence the name `weakprng'.
 *
 * The state consists of a key, a 64-bit counter, and a 64-byte buffer
 * of output, from which 32-bit words are disbursed one by one on
 * request and which is refilled as necessary.  Since the buffer is
 * only ever filled when there is demand for at least a single word,
 * the last word would be unused outside the internal computations.
 * Rather than leave it unused and expand the size of the state to
 * count the number of buffered words remaining, we use the unused
 * word in the buffer to count that.
 */

CTASSERT(crypto_core_OUTPUTBYTES ==
    sizeof(((struct crypto_weakprng *)0)->buffer));
CTASSERT(crypto_core_KEYBYTES ==
    sizeof(((struct crypto_weakprng *)0)->key));
CTASSERT(crypto_core_INPUTBYTES ==
    sizeof(((struct crypto_weakprng *)0)->nonce));

int
crypto_weakprng_selftest(void)
{
	static const uint8_t seed[crypto_weakprng_SEEDBYTES] = {0};
	struct crypto_weakprng P;
	uint64_t v;
	int status;

	status = crypto_core_selftest();
	if (status != 0)
		return status;

	crypto_weakprng_seed(&P, seed);
	v = crypto_weakprng_64(&P);
	if (v != UINT64_C(0x42fe0c0eb8fd7b38))
		return -1;

	return 0;
}

void
crypto_weakprng_seed(struct crypto_weakprng *P, const void *seed)
{

	memset(P, 0, sizeof(*P));
	CTASSERT(sizeof P->key == crypto_weakprng_SEEDBYTES);
	memcpy(P->key, seed, crypto_weakprng_SEEDBYTES);
}

uint32_t
crypto_weakprng_32(struct crypto_weakprng *P)
{
	const unsigned ii = arraycount(P->buffer) - 1;
	uint32_t v;

	/* If there are any 32-bit words left in the buffer, get one.  */
	if (P->buffer[ii]) {
		assert(P->buffer[ii] < 16);
		v = le32dec(&P->buffer[--P->buffer[ii]]);
		goto out;
	}

	/* Otherwise, generate a new block of output.  */
	crypto_core((uint8_t *)P->buffer, (const uint8_t *)P->nonce,
	    (const uint8_t *)P->key, crypto_core_constant32);

	/*
	 * Increment the 64-bit nonce.  Overflow is not a concern: if
	 * we generated a block every nanosecond, it would take >584
	 * years to reach 2^64.
	 */
	le32enc(&P->nonce[0], 1 + le32dec(&P->nonce[0]));
	if (le32dec(&P->nonce[0]) == 0)
		le32enc(&P->nonce[1], 1 + le32dec(&P->nonce[1]));

	/*
	 * Extract the last 32-bit word and use its place to count the
	 * remaining 32-bit words.
	 */
	v = le32dec(&P->buffer[ii]);
	P->buffer[ii] = ii;

out:	return v;
}

uint64_t
crypto_weakprng_64(struct crypto_weakprng *P)
{
	uint32_t lo, hi;

	hi = crypto_weakprng_32(P);
	lo = crypto_weakprng_32(P);

	return ((uint64_t)hi << 32) | (uint64_t)lo;
}

void
crypto_weakprng_buf(struct crypto_weakprng *P, void *buf, size_t len)
{
	uint8_t *p = (uint8_t *)buf;
	uint32_t u32;
	unsigned n32, n8;

	/* Fill as many full 32-bit words as we can.  */
	n32 = len / 4;
	while (n32--) {
		u32 = crypto_weakprng_32(P);
		le32enc(p, u32);
		p += 4;
	}

	/* Fill a partial 32-bit word if necessary.  */
	n8 = len % 4;
	if (n8) {
		u32 = crypto_weakprng_32(P);
		while (n8--) {
			*p++ = u32 & 0xff;
			u32 >>= 8;
		}
	}
}

uintmax_t
crypto_weakprng_below(struct crypto_weakprng *P, uintmax_t bound)
{
	uintmax_t rle, r, minimum;

	/*
	 * Let k be the number of bits in uintmax_t and m = bound.  If
	 * we chose an integer uniformly at random from [0, 2^k), and
	 * reduced it modulo m, values in [0, 2^k mod m) would have
	 * one more representative than values in [2^k mod m, 2^k).
	 * So we first reject values in [0, 2^k mod m) before reducing
	 * modulo m, to avoid this `modulo bias'.  Note that
	 *
	 *	2^k mod m = 2^k mod m - 0
	 *	  = 2^k mod m - m mod m
	 *	  = (2^k - m) mod m,
	 *
	 * which is what (-bound) % bound computes, in k-bit uintmax_t
	 * arithmetic.
	 *
	 * The probability of rejection is never more than 50%, which is
	 * approached only when bound approaches 2^(k-1) from above.
	 */
	minimum = (-bound) % bound;

	do {
		crypto_weakprng_buf(P, &rle, sizeof rle);
		r = lemaxdec(&rle);
	} while (r < minimum);

	return r % bound;
}
