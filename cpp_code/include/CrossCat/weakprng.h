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

#ifndef	WEAKPRNG_H
#define	WEAKPRNG_H

#include <stddef.h>
#include <stdint.h>

struct crypto_weakprng {
	uint32_t buffer[16];
	uint32_t key[8];
	uint32_t nonce[4];
};

#define	crypto_weakprng_SEEDBYTES	32

void		crypto_weakprng_seed(struct crypto_weakprng *, const void *);
uint32_t	crypto_weakprng_32(struct crypto_weakprng *);
uint64_t	crypto_weakprng_64(struct crypto_weakprng *);
void		crypto_weakprng_buf(struct crypto_weakprng *, void *, size_t);
uintmax_t	crypto_weakprng_below(struct crypto_weakprng *, uintmax_t);

int		crypto_weakprng_selftest(void);

#endif	/* WEAKPRNG_H */
