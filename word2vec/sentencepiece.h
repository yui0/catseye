//---------------------------------------------------------
//	Whiz (Japanese Input Method Engine)
//
//		©2022-2023 Yuichiro Nakada
//---------------------------------------------------------

#define handle_error(msg) \
  do { perror(msg); exit(EXIT_FAILURE); } while (0)

#ifndef WORDMAX
#define WORDMAX		40
#endif
#ifndef DICSIZE
#define DICSIZE		8192
#endif

int utf8_bytesize(unsigned char code)
{
	int size = 1;
	if (0x80 & code) { /* 1バイト文字以外 */
		for (int i=2; i<=8; i++) {
			code <<= 1;
			if (!(0x80 & code)) break;
			size++;
		}
	}
	return size;
}

int utf8_strlen(const char* str)
{
	int c, i, ix, q;
	for (q=0, i=0, ix=strlen(str); i<ix; i++, q++) {
		c = (unsigned char) str[i];
		if      (c>=0   && c<=127) i+=0;
		else if ((c & 0xE0) == 0xC0) i+=1;
		else if ((c & 0xF0) == 0xE0) i+=2;
		else if ((c & 0xF8) == 0xF0) i+=3;
		//else if (($c & 0xFC) == 0xF8) i+=4; // 111110bb //byte 5, unnecessary in 4 byte UTF-8
		//else if (($c & 0xFE) == 0xFC) i+=5; // 1111110b //byte 6, unnecessary in 4 byte UTF-8
		else return 0;//invalid utf8
	}
	return q;
}

ht* read_dic(const char *name)
{
	// read dic
	int num = 0;
	int count;
	ht* dic = ht_create();
	if (!dic) handle_error("out of memory");

	FILE *fp = fopen(name, "r");
	fscanf(fp, "%d", &count);
	printf("%d words\n", count);
	while (fscanf(fp, "%d", &count) != EOF) {
		char word[WORDMAX];
		fscanf(fp, "%s", word);
//		printf("%s ", word);

		int* pcount = malloc(sizeof(int)*2);
		if (pcount == NULL) handle_error("out of memory");
		*pcount = count;
		pcount[1] = num++;
		ht_set(dic, word, pcount);
	}
//	printf("\n\n");
	fclose(fp);

	return dic;
}

char* split_word(char* str, ht *dic)
{
	int len = strlen(str);
	char* word = (char*)malloc(sizeof(char) * len);
	char* result = (char*)malloc(sizeof(char) * len*2);
	result[0] = 0;
	for (int i=0; i<len; /*i+=utf8_bytesize(str[i])*/) {
		int l = utf8_bytesize(str[i]);
		for (int n=strlen(&str[i]); n>0; n--) {
			memcpy(word, &str[i], n);
			word[n] = 0;

			const void* value = ht_get(dic, word);
			if (value != NULL) {
//				printf("%s ", word);
/*				strcat(result, word);
				strcat(result, " ");*/
				if (n>0) l = n;
				break;
			}
		}
		strcat(result, word);
		strcat(result, " ");

		i += l;
	}
	free(word);
	return result;
}

size_t* word2int(char* str, ht *dic)
{
	int len = strlen(str);
	char* word = (char*)alloca(sizeof(char) * len);
	size_t* result = (size_t*)malloc(sizeof(size_t) * utf8_strlen(str));
	int m = 0;
	for (int i=0; i<len; /*i+=utf8_bytesize(str[i])*/) {
		int l = utf8_bytesize(str[i]);
		for (int n=strlen(&str[i]); n>0; n--) {
			memcpy(word, &str[i], n);
			word[n] = 0;

			size_t idx = ht_get_index(dic, word);
			if (idx!=SIZE_MAX) {
//				result[m++] = idx;
				int *v = dic->entries[idx].value;
				result[m++] = v[1];
				if (n>0) l = n;
				break;
			}
		}

		i += l;
	}
//	free(word);
	result[m] = SIZE_MAX;
	return result;
}
