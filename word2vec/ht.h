/* Simple hash table implemented in C.
 *	©2023 Yuichiro Nakada
 *
 * This software is released under the MIT License.
 * https://github.com/benhoyt/ht
 * */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#ifndef _HT_H
#define _HT_H

// Hash table structure: create with ht_create, free with ht_destroy.
typedef struct ht ht;

// Hash table iterator: create with ht_iterator, iterate with ht_next.
typedef struct {
	const char* key;  // current key
	void* value;      // current value

	// Don't use these fields directly.
	ht* _table;       // reference to hash table being iterated
	size_t _index;    // current index into ht._entries
} hti;

#endif // _HT_H

// Hash table entry (slot may be filled or empty).
typedef struct {
	const char* key;  // key is NULL if this slot is empty
	void* value;
} ht_entry;

// Hash table structure: create with ht_create, free with ht_destroy.
struct ht {
	ht_entry* entries;  // hash slots
	size_t capacity;    // size of _entries array
	size_t length;      // number of items in hash table
};

#define INITIAL_CAPACITY 16  // must not be zero

ht* ht_create(void)
{
	// Allocate space for hash table struct.
	ht* table = malloc(sizeof(ht));
	if (table == NULL) {
		return NULL;
	}
	table->length = 0;
	table->capacity = INITIAL_CAPACITY;

	// Allocate (zero'd) space for entry buckets.
	table->entries = calloc(table->capacity, sizeof(ht_entry));
	if (table->entries == NULL) {
		free(table); // error, free table before we return!
		return NULL;
	}
	return table;
}

void ht_destroy(ht* table)
{
	// First free allocated keys.
	for (size_t i = 0; i < table->capacity; i++) {
		free((void*)table->entries[i].key);
	}

	// Then free entries array and table itself.
	free(table->entries);
	free(table);
}

#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

// Return 64-bit FNV-1a hash for key (NUL-terminated). See description:
// https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function
static uint64_t hash_key(const char* key)
{
	uint64_t hash = FNV_OFFSET;
	for (const char* p = key; *p; p++) {
		hash ^= (uint64_t)(unsigned char)(*p);
		hash *= FNV_PRIME;
	}
	return hash;
}

void* ht_get(ht* table, const char* key)
{
	// AND hash with capacity-1 to ensure it's within entries array.
	uint64_t hash = hash_key(key);
	size_t index = (size_t)(hash & (uint64_t)(table->capacity - 1));

	// Loop till we find an empty entry.
	while (table->entries[index].key != NULL) {
		if (strcmp(key, table->entries[index].key) == 0) {
			// Found key, return value.
			return table->entries[index].value;
		}
		// Key wasn't in this slot, move to next (linear probing).
		index++;
		if (index >= table->capacity) {
			// At end of entries array, wrap around.
			index = 0;
		}
	}
	return NULL;
}

size_t ht_get_index(ht* this, const char* key)
{
	// AND hash with capacity-1 to ensure it's within entries array.
	uint64_t hash = hash_key(key);
	size_t index = (size_t)(hash & (uint64_t)(this->capacity - 1));

	// Loop till we find an empty entry.
	while (this->entries[index].key != NULL) {
		if (strcmp(key, this->entries[index].key) == 0) {
			// Found key, return value.
//			printf("found:%zu\n", index);
			return index;
		}
		// Key wasn't in this slot, move to next (linear probing).
		index++;
		if (index >= this->capacity) {
			// At end of entries array, wrap around.
			index = 0;
		}
	}
//	printf("not found:%s\n", key);
	return SIZE_MAX;
}
int ht_del(ht *this, const char *key)
{
	size_t i = ht_get_index(this, key);
	if (i==SIZE_MAX) return 0;
//	free((void*)this->entries[i].key);
	char *p = (char*)this->entries[i].key;
	*p = 0;
	int *v = this->entries[i].value;
	*v = -1;
	this->length--;
	return 1;
}

// Internal function to set an entry (without expanding table).
static const char* ht_set_entry(ht_entry* entries, size_t capacity,
                                const char* key, void* value, size_t* plength)
{
	// AND hash with capacity-1 to ensure it's within entries array.
	uint64_t hash = hash_key(key);
	size_t index = (size_t)(hash & (uint64_t)(capacity - 1));

	// Loop till we find an empty entry.
	while (entries[index].key != NULL) {
		if (entries[index].key[0]==0) break; // add
		if (strcmp(key, entries[index].key) == 0) {
			// Found key (it already exists), update value.
			entries[index].value = value;
			return entries[index].key;
		}
		// Key wasn't in this slot, move to next (linear probing).
		index++;
		if (index >= capacity) {
			// At end of entries array, wrap around.
			index = 0;
		}
	}

	// Didn't find key, allocate+copy if needed, then insert it.
	if (plength != NULL) {
		key = strdup(key);
		if (key == NULL) {
			return NULL;
		}
		(*plength)++;
	}
	entries[index].key = (char*)key;
	entries[index].value = value;
	return key;
}

// Expand hash table to twice its current size. Return true on success,
// false if out of memory.
static bool ht_expand(ht* table)
{
	// Allocate new entries array.
	size_t new_capacity = table->capacity * 2;
	if (new_capacity < table->capacity) {
		return false;  // overflow (capacity would be too big)
	}
	ht_entry* new_entries = calloc(new_capacity, sizeof(ht_entry));
	if (new_entries == NULL) {
		return false;
	}

	// Iterate entries, move all non-empty ones to new table's entries.
	for (size_t i = 0; i < table->capacity; i++) {
		ht_entry entry = table->entries[i];
		if (entry.key != NULL) {
			ht_set_entry(new_entries, new_capacity, entry.key,
			             entry.value, NULL);
		}
	}

	// Free old entries array and update this table's details.
	free(table->entries);
	table->entries = new_entries;
	table->capacity = new_capacity;
	return true;
}

const char* ht_set(ht* table, const char* key, void* value)
{
	assert(value != NULL);
	if (value == NULL) return NULL;

	// If length will exceed half of current capacity, expand it.
	if (table->length >= table->capacity / 2) {
		if (!ht_expand(table)) return NULL;
	}

	// Set entry and update length.
	return ht_set_entry(table->entries, table->capacity, key, value, &table->length);
}

size_t ht_length(ht* table)
{
	return table->length;
}

hti ht_iterator(ht* table)
{
	hti it;
	it._table = table;
	it._index = 0;
	return it;
}

bool ht_next(hti* it)
{
	// Loop till we've hit end of entries array.
	ht* table = it->_table;
	while (it->_index < table->capacity) {
		size_t i = it->_index;
		it->_index++;
		if (table->entries[i].key != NULL) {
			// Found next non-empty item, update iterator key and value.
			ht_entry entry = table->entries[i];
			it->key = entry.key;
			it->value = entry.value;
			return true;
		}
	}
	return false;
}

void ht_print(ht* this)
{
	hti it = ht_iterator(this);
	while (ht_next(&it)) {
		printf("%s %d\n", it.key, *(int*)it.value);
	}
}

#if 0
// Example:
// $ echo 'foo bar the bar bar bar the' | ./demo
// foo 1
// bar 4
// the 2
// 3

void exit_nomem(void)
{
	fprintf(stderr, "out of memory\n");
	exit(1);
}

int main(void)
{
	ht* counts = ht_create();
	if (counts == NULL) {
		exit_nomem();
	}

	// Read next word from stdin (at most 100 chars long).
	char word[101];
	while (scanf("%100s", word) != EOF) {
		// Look up word.
		void* value = ht_get(counts, word);
		if (value != NULL) {
			// Already exists, increment int that value points to.
			int* pcount = (int*)value;
			(*pcount)++;
			continue;
		}

		// Word not found, allocate space for new int and set to 1.
		int* pcount = malloc(sizeof(int));
		if (pcount == NULL) {
			exit_nomem();
		}
		*pcount = 1;
		if (ht_set(counts, word, pcount) == NULL) {
			exit_nomem();
		}
	}

//	ht_del(counts, "the");
//	ht_print(counts);

	// Print out words and frequencies, freeing values as we go.
	hti it = ht_iterator(counts);
	while (ht_next(&it)) {
		printf("%s %d\n", it.key, *(int*)it.value);
		free(it.value);
	}

	// Show the number of unique words.
	printf("Unique words: %d\n", (int)ht_length(counts));

	ht_destroy(counts);
	return 0;
}
#endif
