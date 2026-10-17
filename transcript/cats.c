//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <stdlib.h>
#include <ctype.h>

#ifdef CATS_SPEECH_NATIVE
#include "cats_rnnt.h" /* speech + ONNX + native RNN-T; before legacy alloc macros */
#endif
#include "catseye_llm.h"
/* The optional kana2kanji example is not one of the supplied files.
 * Keep other modes compilable when that separate header is absent. */
#if defined(__has_include)
# if __has_include("catseye_kana.h")
#  include "catseye_kana.h"
#  define CATS_HAVE_KANA 1
# endif
#endif
#ifdef CATS_HAYAMIMI
#include "cats_hayamimi.h"
#endif

cats_llm_model llm_model;

void error_usage(char *cmd)
{
	fprintf(stderr, "Usage:   %s [options]\n", cmd);
	fprintf(stderr, "Example: %s -c model.bin -n 256 -p \"Once upon a time\"\n", cmd);
	fprintf(stderr, "kana2kanji:\n");
	fprintf(stderr, "  %s -m ../kana2kanji/out/model_best.gguf -t 0 -n 32 -p \"<h2k>とうきょう<sep>\"\n", cmd);
	fprintf(stderr, "  %s -m models/model_best.gguf -t 0 -n 32 -p \"<k2h>東京<sep>\"\n", cmd);
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -m <string> GGUF / GGML model (kana2kanji: use .gguf; vocab is embedded)\n");

	fprintf(stderr, "  -c <string> checkpoint file (.bin; needs -z tokenizer.bin)\n");
	fprintf(stderr, "  -z <string> optional path to custom tokenizer (for -c only)\n");

	fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0 (kana2kanji: use 0)\n");
	fprintf(stderr, "  -k <float>  k value in top-k sampling in [0,1], default 0.001\n");
	fprintf(stderr, "  -x <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
	fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
	fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
	fprintf(stderr, "  -p <string> input prompt\n");

	fprintf(stderr, "  -a <string> mode: generate|chat|kana|asr|fbank, default: generate\n");
#ifdef CATS_SPEECH_NATIVE
	fprintf(stderr, "  -a cpu-info: report runtime SIMD (CATS_SIMD=scalar|avx|avx2|avx512|auto)\n");
	fprintf(stderr, "  -a fbank -w FILE.wav [-o FILE.cfb] (native C, NO Sherpa/ONNX)\n");
	fprintf(stderr, "  -a eg2-features -w FILE.wav -o features.eg2f (EmbeddingGemma 2 log-mel [3000,128])\n");
	fprintf(stderr, "  -a eg2-decode -m scores.eg2s -z vocab.txt [-L ja] (word-reader CTC output)\n");
	fprintf(stderr, "  -a eg2-subsample -m subsampler.eg2c -w WAV -o hidden.eg2h (Conv2D ONLY)\n");
	fprintf(stderr, "  -a eg2-layer -m layer_00.eg2l -p hidden.eg2h -o result.eg2h (one Conformer layer)\n");
	fprintf(stderr, "  -a eg2-conformer -m layers_directory -p hidden.eg2h -o result.eg2h (12 layers)\n");
	fprintf(stderr, "  -a eg2-project -m projection.eg2p -p hidden.eg2h -o tokens.eg2t (512-d Reader input)\n");
	fprintf(stderr, "  -a eg2-encode -m model_dir -w speech.wav -o tokens.eg2t (subsampler+12 layers+projection)\n");
	fprintf(stderr, "  -a eg2-native-asr -m models -w speech.wav [-L ja|en] (bundled encoder.eg2m + language.eg2m)\n");
	fprintf(stderr, "      -G auto|off|N  GPU GEMM (OpenGL ES 3.1 compute; N = EGL device index; or CATS_GPU=); needs a GPU build\n");
	fprintf(stderr, "      -j N threads (default: physical cores; or OMP_NUM_THREADS); -m models_fp16 from eg2_bundle.py --to-fp16\n");
	fprintf(stderr, "  -a eg2-asr -m encoder.onnx -D reader.onnx -z vocab.txt -w input.wav\n");
    fprintf(stderr, "  -a eg2-asr -m encoder.onnx -R reader.eg2r -Q table.eg2q -z vocab.txt -w WAV\n");
    fprintf(stderr, "  -a eg2-reader -m reader.eg2r -p soft_tokens.eg2t -Q table.eg2q -z vocab.txt [-L ja]\n");
	fprintf(stderr, "      [-o scores.eg2s] (EXPERIMENTAL; native ONNX subset, real models unverified)\n");
	fprintf(stderr, "      [-Q word_table.eg2q -g LOGIT_SCALE] optional INT8 cosine retrieval\n");
	fprintf(stderr, "  -a eg2-quantize -m table.eg2v -o table.eg2q (streaming 768-d word table)\n");
	fprintf(stderr, "            Extract 80-bin log-mel features; NOT transcription\n");
	fprintf(stderr, "  -a tokens -z tokens.txt -p 1,2,3 (byte-BPE IDs -> UTF-8)\n");
	fprintf(stderr, "  -a onnx-info -m MODEL.onnx (list operators, no runtime)\n");
	fprintf(stderr, "  -a onnx-graph -m MODEL.onnx (operators and input/output shapes)\n");
	fprintf(stderr, "  -a onnx-check -m MODEL.onnx (strict preflight; nonzero exit if blocked)\n");
	fprintf(stderr, "  -T trace.tsv for onnx-run, onnx-run-compact or asr-native (per-node trace)\n");
	fprintf(stderr, "  -F profile.tsv for onnx-run, onnx-run-compact or asr-native (hotspot timing)\n");
	fprintf(stderr, "  -S 1,1,4 specifies input shape for onnx-run modes (default 1-D)\n");
	fprintf(stderr, "  -a asr-native -m encoder.onnx -D decoder.onnx -J joiner.onnx\n");
	fprintf(stderr, "       -z tokens.txt -w input.wav [-C context] [-B beam] [-V]\n");
	fprintf(stderr, "       -V energy VAD + 1s real audio preroll (NOT Silero VAD)\n");
	fprintf(stderr, "  -a onnx-tensor -m MODEL.onnx -p NAME (read initializer values)\n");
	fprintf(stderr, "  -a onnx-run -m SMALL.onnx -p 1,2,3 (single vector input only)\n");
	fprintf(stderr, "  -a onnx-run-compact -m SMALL.onnx -p 1,2,3 (reclaim intermediates)\n");
#endif
#ifdef CATS_HAYAMIMI
	fprintf(stderr, "Hayamimi minimal C audio-to-text (no Python):\n");
	fprintf(stderr, "  -a asr -w FILE.wav [-d hayamimi/models] [-l ja|zh|ko|yue|auto|en|fr|de|es|it|pt]\n");
	fprintf(stderr, "  -a asr -r [-P] < input.s16le   (mono, 16000Hz; energy VAD)\n");
	fprintf(stderr, "  -b PATH    libsherpa-onnx-c-api.so (sherpa-onnx 1.13.6 only)\n");
	fprintf(stderr, "  -j N       CPU threads (default 4)\n");
	fprintf(stderr, "  -d DIR     downloaded Hayamimi models directory (default models)\n");
	fprintf(stderr, "  -l LANG    fixed ASR route; auto=SenseVoice-only 5 languages\n");
	fprintf(stderr, "  -w FILE    16-bit PCM mono WAV file\n");
	fprintf(stderr, "  -r         stdin raw signed 16-bit little-endian PCM, 16k mono\n");
	fprintf(stderr, "  -P         show provisional transcriptions (raw mode only)\n");
#else
	fprintf(stderr, "  (for -a asr, rebuild with -DCATS_HAYAMIMI)\n");
#endif
	fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
	// default parameters
	char *model = NULL;
	char *checkpoint_path = "models/stories42M.bin";
	char *tokenizer_path = "models/tokenizer.bin";
	int tokenizer_given = 0;	// -z was passed explicitly
	char *prompt = NULL;		// prompt string
	char *mode = "generate";	// generate|chat
	char *system_prompt = NULL;	// the (optional) system prompt to use in chat mode
	const char *native_eg2_reader = NULL, *native_decoder = NULL, *native_joiner = NULL, *native_trace = NULL, *native_shape = NULL, *native_profile = NULL, *native_q8_table = NULL;
	float native_eg2_logit_scale = 0;
    int eg2_japanese = 0;
	int native_context = 0, native_beam = 4, native_vad = 0;
	const char *asr_wav = NULL, *asr_root = "models", *asr_lang = "ja", *asr_lib = NULL, *fbank_out = NULL;
	int asr_threads = 4, asr_raw = 0, asr_partial = 0, threads_given = 0;
	const char *gpu_spec = NULL;	// -G auto|off|N (needs a -DCATS_GPU build)
	cats_llm_model *m = &llm_model;
	m->steps = 512;			// max number of steps to run for, 0: use seq_len
	m->temperature = 1.0;		// 0.0 = greedy deterministic. 1.0 = original. don't set higher
	m->topp = 0.9;			// top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
	m->topk = 0.001;
	rng_seed = 0;			// seed rng with time by default
	//srand((unsigned int)time(NULL));

	for (int i=1; i<argc; i++) {
		switch (argv[i][0]) {
		case '-':
			switch (argv[i][1]) {
			case 'm': if (i+1 < argc) { model = argv[++i]; }		break;

			case 'c': if (i+1 < argc) { checkpoint_path = argv[++i]; }	break;
			case 'z': if (i+1 < argc) { tokenizer_path = argv[++i]; tokenizer_given = 1; }	break;

			case 't': if (i+1 < argc) { m->temperature = atof(argv[++i]); }	break;
			case 'k': if (i+1 < argc) { m->topk = atof(argv[++i]); }	break;
			case 'x': if (i+1 < argc) { m->topp = atof(argv[++i]); }	break;
			case 's': if (i+1 < argc) { rng_seed = atoi(argv[++i]); }	break;
			case 'n': if (i+1 < argc) { m->steps = atoi(argv[++i]); }	break;
			case 'D': if (i+1 < argc) { native_decoder = argv[++i]; } else return 1; break;
			case 'J': if (i+1 < argc) { native_joiner = argv[++i]; } else return 1; break;
			case 'C': if (i+1 < argc) { native_context = atoi(argv[++i]); } else return 1; break;
			case 'B': if (i+1 < argc) { native_beam = atoi(argv[++i]); } else return 1; break;
			case 'Q': if (i+1 < argc) { native_q8_table = argv[++i]; } else return 1; break;
            case 'L': if(i+1<argc){const char*lang=argv[++i];if(!strcmp(lang,"ja"))eg2_japanese=1;else if(!strcmp(lang,"en"))eg2_japanese=0;else {fprintf(stderr,"-L needs en or ja\n");return 1;}}else return 1;break;
            case 'R': if (i+1 < argc) { native_eg2_reader = argv[++i]; } else return 1; break;
			case 'g': if (i+1 < argc) { native_eg2_logit_scale = strtof(argv[++i],NULL); } else return 1; break;
			case 'V': native_vad = 1; break;
			case 'T': if (i+1 < argc) { native_trace = argv[++i]; } else return 1; break;
			case 'F': if (i+1 < argc) { native_profile = argv[++i]; } else return 1; break;
			case 'S': if (i+1 < argc) { native_shape = argv[++i]; } else return 1; break;
			case 'p': if (i+1 < argc) { prompt = argv[++i]; }		break;

			case 'a': if (i+1 < argc) { mode = argv[++i]; }			break;
			case 'y': if (i+1 < argc) { system_prompt = argv[++i]; }	break;
			case 'w': if (i+1 < argc) { asr_wav = argv[++i]; } else return 1; break;
			case 'o': if (i+1 < argc) { fbank_out = argv[++i]; } else return 1; break;
			case 'd': if (i+1 < argc) { asr_root = argv[++i]; } else return 1; break;
			case 'l': if (i+1 < argc) { asr_lang = argv[++i]; } else return 1; break;
			case 'b': if (i+1 < argc) { asr_lib = argv[++i]; } else return 1; break;
				case 'G': if (i+1 < argc) { gpu_spec = argv[++i]; } else return 1; break;
			case 'j': if (i+1 < argc) { asr_threads = atoi(argv[++i]); threads_given = 1; } else return 1; break;
			case 'r': asr_raw = 1; break;
			case 'P': asr_partial = 1; break;
			default: fprintf(stderr,  "Invalid option: %s\n", argv[i]);	return 1;
			} break;
		default:
			error_usage(argv[0]);
		}
	}
#ifdef CATS_SPEECH_NATIVE
#ifdef _OPENMP
	/* -j N also sets the EmbeddingGemma 2 thread count (default: all cores, or OMP_NUM_THREADS). */
	if (threads_given && asr_threads > 0) omp_set_num_threads(asr_threads);
	else if (!getenv("OMP_NUM_THREADS") && cs_eg2_physical_threads() > 0) omp_set_num_threads(cs_eg2_physical_threads());
#endif
#ifdef CATS_GPU
	if (!strncmp(mode, "eg2-", 4) && !cs_gpu_init(gpu_spec)) fprintf(stderr, "cats: GPU GEMM on %s\n", cs_gpu_name());
#else
	(void)gpu_spec;
#endif
    if(strcmp(mode,"cpu-info")==0){
      printf("Cat's eye native ONNX SIMD: %s\n",cats_cpu_simd_name());
      printf("INT8 VNNI: %s\n",cats_cpu_has_vnni()?"AVX-512 VNNI enabled":"AVX2/scalar fallback");
      printf("Override via CATS_SIMD=scalar|avx|avx2|avx512|auto (always CPU-checked).\n");
      return 0;
    }
	/* EmbeddingGemma 2 architecture: audio frontend, word CTC decoding and an
	 * experimental, explicitly checked native ONNX bridge. */
	if(strcmp(mode,"eg2-features")==0){
		if(!asr_wav||!fbank_out){fprintf(stderr,"eg2-features: -w WAV -o FILE required\n");return 1;}
		cs_wave wave={0};float*rs=NULL;size_t n=0;
		if(cs_wave_read(asr_wav,&wave)){fprintf(stderr,"eg2-features: invalid WAV\n");return 1;}
		if(wave.rate==16000){rs=wave.pcm;n=wave.count;wave.pcm=NULL;}
		else if(cs_resample(wave.pcm,wave.count,wave.rate,16000,&rs,&n)){
			cs_wave_free(&wave);fprintf(stderr,"eg2-features: resample failed\n");return 1;
		}
		cs_wave_free(&wave);
		if(n>480000){fprintf(stderr,"eg2-features: single-window mode accepts <=30s (eg2-asr handles overlap)\n");free(rs);return 1;}
		cs_eg2_frontend *front=malloc(sizeof(*front));
		float*features=malloc((size_t)CS_EG2_BUCKET*CS_EG2_MEL*sizeof(float));
		if(!front||!features){free(front);free(features);free(rs);return 1;}
		size_t frames=0;cs_eg2_frontend_init(front);
		int status=cs_eg2_features(front,rs,n,features,&frames);
		if(!status)status=cs_eg2_write_features(fbank_out,features,frames);
		if(status)fprintf(stderr,"eg2-features: failed to extract/write features\n");
		else printf("EmbeddingGemma2: %zu valid frames / 3000 padded, 128 bins, %zu tokens\n",frames,(frames+3)/4);
		free(front);free(features);free(rs);return status?1:0;
	}
	if(strcmp(mode,"eg2-subsample")==0||strcmp(mode,"eg2-encode")==0){
		if(!model||!asr_wav||!fbank_out){
			fprintf(stderr,"eg2-subsample: -m subsample.eg2c -w WAV -o hidden.eg2h required\n");return 1;
		}
		cs_eg2_subsample sub={0};cs_wave wave={0};
		cs_eg2_frontend *front=NULL;float *rs=NULL,*mel=NULL,*out=NULL;
		size_t ns=0,nframes=0,tokens=0;int rc=1;
        int full_encoder=!strcmp(mode,"eg2-encode");
        char subsampler_file[4096];const char*stem_path=model;
        if(full_encoder){
            if(snprintf(subsampler_file,sizeof(subsampler_file),"%s/subsampler.eg2c",model)>=(int)sizeof(subsampler_file))goto subsample_done;
            stem_path=subsampler_file;
        }
		if(cs_eg2_subsample_load(&sub,stem_path)){
			fprintf(stderr,"eg2-encode: incompatible or damaged EG2C weights\n");goto subsample_done;
		}
		if(cs_wave_read(asr_wav,&wave))goto subsample_done;
		if(wave.rate==16000){rs=wave.pcm;ns=wave.count;wave.pcm=NULL;}
		else if(cs_resample(wave.pcm,wave.count,wave.rate,16000,&rs,&ns))goto subsample_done;
		if(ns>480000||ns<320){fprintf(stderr,"eg2-subsample: WAV must be >=20ms and <=30s\n");goto subsample_done;}
		front=malloc(sizeof(*front));
		mel=malloc((size_t)CS_EG2_BUCKET*CS_EG2_MEL*sizeof(float));
		out=malloc((size_t)CS_EG2_TOKENS*CS_EG2_SH*sizeof(float));
		if(!front||!mel||!out)goto subsample_done;
		cs_eg2_frontend_init(front);
		if(cs_eg2_features(front,rs,ns,mel,&nframes)||
		   cs_eg2_subsample_run(&sub,mel,nframes,nframes,out,(size_t)CS_EG2_TOKENS*CS_EG2_SH,&tokens)){
			fprintf(stderr,"eg2-encode: native Conv2D inference failed\n");goto subsample_done;
		}
        if(!full_encoder){
            if(cs_eg2_hidden_write(fbank_out,out,tokens))goto subsample_done;
            printf("EmbeddingGemma2 subsampler: %zu frames -> %zu x 1024 hidden states (NOT final ASR)\n",nframes,tokens);
        }else{
            int okay=1;
            for(int i=0;i<12;i++){
                char layer_path[4096];
                if(snprintf(layer_path,sizeof(layer_path),"%s/layer_%02d.eg2l",model,i)>=(int)sizeof(layer_path)){okay=0;break;}
                cs_eg2_layer layer={0};
                if(cs_eg2_layer_load(&layer,layer_path)||layer.d!=1024||layer.ff!=4096||layer.heads!=8||layer.kernel!=5||
                   cs_eg2_layer_run(&layer,out,tokens,tokens))okay=0;
                cs_eg2_layer_free(&layer);
                if(!okay){fprintf(stderr,"eg2-encode: failed Conformer layer %d\n",i);break;}
            }
            if(!okay)goto subsample_done;
            char project_path[4096];
            if(snprintf(project_path,sizeof(project_path),"%s/projection.eg2p",model)>=(int)sizeof(project_path))goto subsample_done;
            cs_eg2_projection projection={0};
            float *soft=malloc(tokens*512*sizeof(float));
            if(!soft||cs_eg2_projection_load(&projection,project_path)||
               cs_eg2_projection_run(&projection,out,tokens,soft)||
               cs_eg2_soft_write(fbank_out,soft,tokens))okay=0;
            cs_eg2_projection_free(&projection);free(soft);
            if(!okay){fprintf(stderr,"eg2-encode: failed final projection or EG2T output\n");goto subsample_done;}
            printf("EmbeddingGemma2 native: %zu log-mel frames -> %zu x 512 soft tokens (NOT yet transcription)\n",nframes,tokens);
        }
		rc=0;
subsample_done:
		cs_wave_free(&wave);free(rs);free(front);free(mel);free(out);
		cs_eg2_subsample_free(&sub);return rc;
	}
	if(strcmp(mode,"eg2-layer")==0||strcmp(mode,"eg2-conformer")==0){
        if(!model||!prompt||!fbank_out){
            fprintf(stderr,"eg2-layer: -m weight.eg2l -p input.eg2h -o output.eg2h required\n");return 1;
        }
        int many=!strcmp(mode,"eg2-conformer"),count=many?12:1,rc=1;
        float*hidden=NULL;size_t t=0;
        if(cs_eg2_hidden_read(prompt,&hidden,&t)){
            fprintf(stderr,"eg2-layer: invalid 1024-wide EG2H input\n");return 1;
        }
        int ok=1;
        for(int i=0;i<count;i++){
            char path[4096];const char*filename=model;
            if(many){
                if(snprintf(path,sizeof(path),"%s/layer_%02d.eg2l",model,i)>=(int)sizeof(path)){ok=0;break;}
                filename=path;
            }
            cs_eg2_layer layer={0};
            if(cs_eg2_layer_load(&layer,filename)||layer.d!=1024||layer.ff!=4096||layer.heads!=8||layer.kernel!=5){
                fprintf(stderr,"eg2-layer: damaged/incompatible weights %s\n",filename);
                cs_eg2_layer_free(&layer);ok=0;break;
            }
            if(cs_eg2_layer_run(&layer,hidden,t,t)){
                fprintf(stderr,"eg2-layer: Conformer computation failed at %s\n",filename);
                cs_eg2_layer_free(&layer);ok=0;break;
            }
            cs_eg2_layer_free(&layer);
        }
        if(ok&&!cs_eg2_hidden_write(fbank_out,hidden,t)){
            printf("Conformer: %d layer(s), %zu x 1024 hidden -> %s (not yet soft tokens)\n",count,t,fbank_out);
            rc=0;
        }
        free(hidden);return rc;
    }
	if(strcmp(mode,"eg2-project")==0){
        if(!model||!prompt||!fbank_out){
            fprintf(stderr,"eg2-project: -m projection.eg2p -p hidden.eg2h -o tokens.eg2t required\n");return 1;
        }
        cs_eg2_projection proj={0};float *hidden=NULL,*soft=NULL;size_t n=0;
        int rc=1;
        if(cs_eg2_hidden_read(prompt,&hidden,&n)||cs_eg2_projection_load(&proj,model)){
            fprintf(stderr,"eg2-project: invalid EG2P or EG2H input\n");goto project_done;
        }
        soft=malloc(n*512*sizeof(float));if(!soft)goto project_done;
        if(cs_eg2_projection_run(&proj,hidden,n,soft)||cs_eg2_soft_write(fbank_out,soft,n)){
            fprintf(stderr,"eg2-project: 1024->512 final projection failed\n");goto project_done;
        }
        printf("Gemma4 audio projection: %zu x 1024 -> %zu x 512 soft tokens (%s)\n",n,n,fbank_out);
        rc=0;
project_done:free(hidden);free(soft);cs_eg2_projection_free(&proj);return rc;
    }
	if(strcmp(mode,"eg2-decode")==0){
		if(!model){fprintf(stderr,"eg2-decode: -m scores.eg2s required\n");return 1;}
		cs_eg2_score*scores=NULL;size_t n=0;cs_eg2_vocab vocab={0};
		if(cs_eg2_read_scores(model,&scores,&n)||cs_eg2_vocab_load(&vocab,tokenizer_path)){
			fprintf(stderr,"eg2-decode: invalid score file or vocabulary\n");
			free(scores);cs_eg2_vocab_free(&vocab);return 1;
		}
		size_t cap=n>32768?4u*1024u*1024u:n*128+1;
		char*buffer=malloc(cap);
		int result=buffer?cs_eg2_ctc_decode_lang(scores,n,&vocab,buffer,cap,eg2_japanese):-1;
		if(result>=0)puts(buffer);else fprintf(stderr,"eg2-decode: corrupt scores or output too large\n");
		free(buffer);free(scores);cs_eg2_vocab_free(&vocab);return result<0;
	}
	if(strcmp(mode,"eg2-quantize")==0){
		if(!model||!fbank_out){fprintf(stderr,"eg2-quantize: -m TABLE.eg2v -o TABLE.eg2q required\n");return 1;}
		if(cs_eg2_quantize_table(model,fbank_out)){
			fprintf(stderr,"eg2-quantize: invalid EG2V table or write failure\n");return 1;
		}
		printf("EmbeddingGemma 2: quantized text word table -> %s\n",fbank_out);
		return 0;
	}
    if(strcmp(mode,"eg2-reader")==0){
        if(!model||!prompt||!native_q8_table){
            fprintf(stderr,"eg2-reader: -m reader.eg2r -p tokens.eg2t -Q table.eg2q -z vocab.txt [-o scores.eg2s]\n");return 1;
        }
        cs_eg2_reader r={0};cs_eg2_q8 q8={0};cs_eg2_vocab v={0};
        float *soft=NULL,*projected=NULL,*blank=NULL;cs_eg2_score*scores=NULL;
        char*output=NULL;size_t n=0;int result=-1;
        if(cs_eg2_reader_load(&r,model)||cs_eg2_soft_read(prompt,&soft,&n)||
           cs_eg2_q8_open(&q8,native_q8_table)||cs_eg2_vocab_load(&v,tokenizer_path)||
           q8.count!=v.count)goto reader_done;
        projected=malloc(n*768*sizeof(float));blank=malloc(n*sizeof(float));
        scores=malloc(n*sizeof(*scores));output=malloc(n*128+1);
        if(!projected||!blank||!scores||!output||
           cs_eg2_reader_run(&r,soft,n,projected,blank))goto reader_done;
        if(cs_eg2_nearest_batch(&q8,projected,768,r.scale,blank,n,scores))goto reader_done;
        if(fbank_out&&cs_eg2_write_scores(fbank_out,scores,n))goto reader_done;
        result=cs_eg2_ctc_decode_lang(scores,n,&v,output,n*128+1,eg2_japanese);
        if(result>=0)puts(output);
reader_done:
        if(result<0)fprintf(stderr,"eg2-reader: invalid model/tokens/table or native computation failed\n");
        cs_eg2_reader_free(&r);cs_eg2_q8_free(&q8);cs_eg2_vocab_free(&v);
        free(soft);free(projected);free(blank);free(scores);free(output);
        return result<0;
    }
    if(strcmp(mode,"eg2-native-asr")==0){
        if(!model)model="models";
        if(!asr_wav){
            fprintf(stderr,"eg2-native-asr: -m model_dir -w WAV [-L ja|en] (auto .eg2m)\n");return 1;
        }
        char bundle_encoder[4096],bundle_reader[4096],bundle_table[4096],bundle_vocab[4096];
        const char *encoder_path=model;
        const char *reader_path=native_eg2_reader,*table_path=native_q8_table,*vocab_path=tokenizer_given?tokenizer_path:NULL;
        /* Auto resolve packages only when the bundled encoder is present.
         * Explicit legacy arguments continue working with unpacked models. */
        if(!reader_path||!table_path||!vocab_path){
            int ok=snprintf(bundle_encoder,sizeof(bundle_encoder),"%s/encoder.eg2m",model)<(int)sizeof(bundle_encoder) &&
                   snprintf(bundle_reader,sizeof(bundle_reader),"%s/%s.eg2m/reader.eg2r",model,eg2_japanese?"ja":"en")<(int)sizeof(bundle_reader) &&
                   snprintf(bundle_table,sizeof(bundle_table),"%s/%s.eg2m/word_table.eg2q",model,eg2_japanese?"ja":"en")<(int)sizeof(bundle_table) &&
                   snprintf(bundle_vocab,sizeof(bundle_vocab),"%s/%s.eg2m/reader_vocab.txt",model,eg2_japanese?"ja":"en")<(int)sizeof(bundle_vocab);
            if(!ok){fprintf(stderr,"eg2-native-asr: model path too long\n");return 1;}
            FILE *probe=fopen(bundle_encoder,"rb");
            if(!probe){fprintf(stderr,"eg2-native-asr: missing %s (or specify legacy -R -Q -z)\n",bundle_encoder);return 1;}
            fclose(probe);
            encoder_path=bundle_encoder;
            if(!reader_path)reader_path=bundle_reader;
            if(!table_path)table_path=bundle_table;
            if(!vocab_path)vocab_path=bundle_vocab;
        }
        char*transcript=malloc(4u*1024u*1024u);if(!transcript)return 1;
        int rc=cs_eg2_native_long_wav(encoder_path,reader_path,table_path,
               vocab_path,asr_wav,transcript,4u*1024u*1024u,fbank_out,eg2_japanese);
        if(!rc)puts(transcript);
        else fprintf(stderr,"eg2-native-asr: invalid weights, audio or inference failure (no transcript emitted)\n");
        free(transcript);
#ifdef CATS_GPU
        cs_gpu_term();
#endif
        return rc!=0;
    }
	if(strcmp(mode,"eg2-asr")==0){
		if(!model||!asr_wav||((native_eg2_reader==NULL)==(native_decoder==NULL))){
            fprintf(stderr,"eg2-asr: -m encoder.onnx and either -D reader.onnx or -R native_reader.eg2r, -z vocab.txt -w WAV required\n");return 1;
		}
		/* Output estimate: bound from WAV size; cap at 4 MiB. */
		char*text=malloc(4u*1024u*1024u);
		if(!text)return 1;
		int rc=cx_eg2_transcribe_wav_ext_lang(model,native_decoder,tokenizer_path,asr_wav,
            text,4u*1024u*1024u,fbank_out,native_q8_table,native_eg2_logit_scale,native_eg2_reader,eg2_japanese);
		if(rc>=0)puts(text);else fprintf(stderr,"eg2-asr: inference failed (model/ONNX support must be validated)\n");
		free(text);return rc<0;
	}
	/* Native C audio -> encoder -> decoder/joiner -> beam -> UTF-8.
	 * This is experimental and must NOT be confused with validated ReazonSpeech.
	 */
	if (strcmp(mode,"asr-native")==0) {
		if(!model||!native_decoder||!native_joiner||!asr_wav){
			fprintf(stderr,"asr-native: requires -m ENCODER -D DECODER -J JOINER -z TOKENS -w WAV\n");return 1;
		}
		cx_native_asr a={0};
		if(cx_native_open(&a,model,native_decoder,native_joiner,tokenizer_path,native_context,native_beam)){
			fprintf(stderr,"asr-native: %s\n",a.error[0]?a.error:"cannot load model set");cx_native_close(&a);return 1;
		}
		if(cx_native_supported(&a,stderr)){
			fprintf(stderr,"asr-native: unsupported operators. Refusing to emit unvalidated transcript.\n");
			cx_native_close(&a);return 1;
		}
        FILE *profile_fp=NULL;
        cx_profiler encoder_profile={0},decoder_profile={0},joiner_profile={0};
        if(native_profile){
          profile_fp=fopen(native_profile,"w");
          if(!profile_fp){perror("asr-native -F");cx_native_close(&a);return 1;}
          cx_profiler_header(profile_fp);
          if(cx_profiler_init(&encoder_profile,&a.encoder.graph,"encoder") ||
             cx_profiler_init(&decoder_profile,&a.decoder.graph,"decoder") ||
             cx_profiler_init(&joiner_profile,&a.joiner.graph,"joiner")){
            fprintf(stderr,"asr-native: profiler allocation failed\n");
            cx_profiler_detach(&a.encoder.graph);cx_profiler_detach(&a.decoder.graph);cx_profiler_detach(&a.joiner.graph);
            cx_profiler_free(&encoder_profile);cx_profiler_free(&decoder_profile);cx_profiler_free(&joiner_profile);
            fclose(profile_fp);cx_native_close(&a);return 1;
          }
        }
        FILE *trace_fp=NULL;
        if(native_trace){
          trace_fp=fopen(native_trace,"w");
          if(!trace_fp){
            perror("asr-native -T");
            cx_profiler_detach(&a.encoder.graph);cx_profiler_detach(&a.decoder.graph);cx_profiler_detach(&a.joiner.graph);
            cx_profiler_free(&encoder_profile);cx_profiler_free(&decoder_profile);cx_profiler_free(&joiner_profile);
            if(profile_fp)fclose(profile_fp);
            cx_native_close(&a);return 1;
          }
          a.encoder.graph.trace=a.decoder.graph.trace=a.joiner.graph.trace=trace_fp;
          a.encoder.graph.trace_name="encoder";
          a.decoder.graph.trace_name="decoder";
          a.joiner.graph.trace_name="joiner";
          fprintf(trace_fp,"graph\tnode\top\toutput\ttype\tshape\telements\tmin\tmax\tmean\tnonfinite\tfnv64\n");
        }
		char text[65536];size_t segs=0;
        int n=native_vad?cx_native_transcribe_wav_segmented(&a,asr_wav,text,sizeof(text),&segs):
                         cx_native_transcribe_wav(&a,asr_wav,text,sizeof(text));
        if(n>=0 && native_vad)fprintf(stderr,"asr-native: %zu VAD segment(s) (energy-based)\n",segs);
		if(n<0)fprintf(stderr,"asr-native: graph execution/decoding failed (not yet validated for actual Zipformer).\n");
		else puts(text);
        int profile_error=0;
        if(profile_fp){
          if(cx_profiler_report(profile_fp,&a.encoder.graph,&encoder_profile) ||
             cx_profiler_report(profile_fp,&a.decoder.graph,&decoder_profile) ||
             cx_profiler_report(profile_fp,&a.joiner.graph,&joiner_profile))profile_error=1;
          if(fclose(profile_fp))profile_error=1;
        }
        cx_profiler_detach(&a.encoder.graph);cx_profiler_detach(&a.decoder.graph);cx_profiler_detach(&a.joiner.graph);
        cx_profiler_free(&encoder_profile);cx_profiler_free(&decoder_profile);cx_profiler_free(&joiner_profile);
        cx_native_close(&a);
        if(trace_fp && fclose(trace_fp))return 1;
        return n<0 || profile_error;
	}
	/* Decode ReazonSpeech's token symbols without loading a neural model. */
	if (strcmp(mode,"tokens")==0) {
		if(!prompt){fprintf(stderr,"tokens: -p token-id-list required\n");return 1;}
		cs_vocab vocab={0};
		if(cs_vocab_load(&vocab,tokenizer_path)){
			fprintf(stderr,"tokens: cannot load vocabulary: %s\n",tokenizer_path);return 1;
		}
		int ids[8192];size_t nids=0;
		const char*p=prompt;int valid=1;
		while(*p){
			if(nids>=sizeof(ids)/sizeof(ids[0])){valid=0;break;}
			char *end=NULL;long id=strtol(p,&end,10);
			if(end==p||id<0||id>INT_MAX){valid=0;break;}
			ids[nids++]=(int)id;
			if(*end==','){p=end+1;if(!*p)valid=0;}
			else if(!*end){p=end;}
			else {valid=0;break;}
			if(!valid)break;
		}
		char out[65536];
		int len=valid?cs_vocab_decode(&vocab,ids,nids,1,out,sizeof(out)):-1;
		if(len>=0)printf("%s\n",out);
		else fprintf(stderr,"tokens: invalid token or insufficient output buffer\n");
		cs_vocab_free(&vocab);
		return len<0;
	}
	/* Read model graph metadata without protobuf / ONNX libraries. */
	if (strcmp(mode,"onnx-info") == 0){
		coi_stats info;
		if(!model||coi_inspect(model,&info)){
			fprintf(stderr,"onnx-info: cannot inspect ONNX model: %s\n",model?model:"(missing -m)");
			return 1;
		}
		coi_print(&info,stdout);
		return 0;
	}
	if (strcmp(mode,"onnx-graph") == 0 || strcmp(mode,"onnx-check") == 0 || strcmp(mode,"onnx-tensor") == 0 || (strcmp(mode,"onnx-run") == 0 || strcmp(mode,"onnx-run-compact") == 0)) {
		cx_graph g;
		if (!model || cx_graph_open(&g,model)) {
			fprintf(stderr,"native ONNX: failed to parse model: %s\n",model?model:"(missing -m)");
			return 1;
		}
        const int running_model=strcmp(mode,"onnx-run")==0 || strcmp(mode,"onnx-run-compact")==0;
        FILE *profile_fp=NULL;
        cx_profiler profiler={0};
        if(native_profile && running_model){
          profile_fp=fopen(native_profile,"w");
          if(!profile_fp){perror("onnx-run -F");cx_graph_close(&g);return 1;}
          cx_profiler_header(profile_fp);
          if(cx_profiler_init(&profiler,&g,"model")){
            fprintf(stderr,"onnx-run: profiler allocation failed\n");
            fclose(profile_fp);cx_graph_close(&g);return 1;
          }
        }
        FILE *trace_fp=NULL;
        if(native_trace && running_model){
          trace_fp=fopen(native_trace,"w");
          if(!trace_fp){
            perror("onnx-run -T");cx_profiler_detach(&g);cx_profiler_free(&profiler);
            if(profile_fp)fclose(profile_fp);
            cx_graph_close(&g);return 1;
          }
          g.trace=trace_fp;g.trace_name="model";
          fprintf(trace_fp,"graph\tnode\top\toutput\ttype\tshape\telements\tmin\tmax\tmean\tnonfinite\tfnv64\n");
        }
		int ret=0;
		if (strcmp(mode,"onnx-graph") == 0) {
			cx_graph_report(&g,stdout);
			cx_graph_interfaces(&g,stdout);
			cx_audit(&g,stdout);
            cx_graph_preflight(&g,stdout);
		} else if (strcmp(mode,"onnx-check") == 0) {
            cx_graph_report(&g,stdout);
            ret=cx_graph_preflight(&g,stdout)?1:0;
        } else if (strcmp(mode,"onnx-run") == 0 || strcmp(mode,"onnx-run-compact") == 0) {
			/* Debug/demo executor, not a production ASR entry point. */
			if(g.ninput!=1||g.noutput<1||!prompt){
				fprintf(stderr,"onnx-run requires one input, one output and -p comma-separated floats\n");ret=1;
			} else {
				float values[4096];int nv=0;
				const char *v=prompt;
				while(*v&&nv<4096){char *end=NULL;float f=strtof(v,&end);
					if(end==v||!isfinite(f)){nv=0;break;}values[nv++]=f;
					if(*end==',')v=end+1;
					else if(*end==0){v=end;break;}
					else{nv=0;break;}
				}
				if(!nv||*v){fprintf(stderr,"onnx-run: invalid input vector\n");ret=1;}
				else{
					cx_env env={0};int64_t dims[CX_DIMS]={0};int nd=1;dims[0]=nv;
                    if(native_shape){
                      nd=0;uint64_t prod=1;
                      const char *cursor=native_shape;
                      while(*cursor && nd<CX_DIMS){
                        char *end=NULL;errno=0;long long value=strtoll(cursor,&end,10);
                        if(errno||end==cursor||value<0||value>4096 ||
                           (value && prod>4096/(uint64_t)value)){nd=-1;break;}
                        prod*=(uint64_t)value;dims[nd++]=(int64_t)value;
                        if(*end==','){cursor=end+1;if(!*cursor){nd=-1;break;}}
                        else if(!*end){cursor=end;break;}
                        else {nd=-1;break;}
                      }
                      if(nd<=0 || *cursor || prod!=(uint64_t)nv){
                        fprintf(stderr,"onnx-run: -S dimensions must multiply to -p values\n");ret=1;
                      }
                    }
					cx_tensor *x=ret?NULL:cx_env_new(&env,g.input[0],dims,nd,
                          g.input_meta[0].type?g.input_meta[0].type:1);
					if(!x)ret=1;
					else{
						memcpy(x->v,values,(size_t)nv*sizeof(float));
						if((strcmp(mode,"onnx-run-compact")==0?cx_graph_run_compact(&g,&env):cx_graph_run(&g,&env)))ret=1;
						else {cx_tensor *y=cx_env_find(&env,g.output[0]);
							if(!y)ret=1;
							else {printf("%s =",g.output[0]);for(uint64_t j=0;j<y->n;j++)printf(" %g",y->v[j]);putchar('\n');}
						}
					}
					cx_env_free(&env);
				}
			}
		} else {
			const cx_weight *w=prompt?cx_find_weight(&g,prompt):NULL;
			if (!w) {fprintf(stderr,"native ONNX: initializer not found: %s\n",prompt?prompt:"(missing -p)");ret=1;}
			else {
				cx_tensor val;
				if (cx_load_weight(&g,w,&val)) {fprintf(stderr,"native ONNX: initializer encoding not supported\n");ret=1;}
				else {printf("%s (%s): %llu values\n",w->name,cx_type(w->type),(unsigned long long)val.n);
					for(uint64_t j=0;j<val.n&&j<32;j++)printf("%s%.9g",j?", ":"",val.v[j]);puts(val.n>32?", ...":"");free(val.v);}
			}
		}
        if(profile_fp){
          if(cx_profiler_report(profile_fp,&g,&profiler))ret=1;
          if(fclose(profile_fp))ret=1;
        }
        cx_profiler_detach(&g);cx_profiler_free(&profiler);
        cx_graph_close(&g);
        if(trace_fp && fclose(trace_fp))ret=1;
		return ret;
	}
	/* Native C-only preprocessing. This is NOT a functional ASR recognizer. */
	if (strcmp(mode,"fbank") == 0) {
		cs_wave wave = {0}; cs_features features = {0};
		if (!asr_wav || cs_wave_read(asr_wav,&wave)) {
			fprintf(stderr,"fbank: cannot read WAV: %s\n",asr_wav?asr_wav:"(missing -w)");
			return 1;
		}
		if (cs_fbank80(wave.pcm,wave.count,wave.rate,&features)) {
			fprintf(stderr,"fbank: feature extraction failed\n"); cs_wave_free(&wave);return 1;
		}
		printf("fbank: %zu frames x %d mel bins, %d Hz (native C)\n",features.frames,features.dim,features.rate);
		int status=0;
		if(fbank_out){status=cs_fbank_write(fbank_out,&features);if(status)fprintf(stderr,"fbank: cannot save %s\n",fbank_out);}
		cs_fbank_free(&features);cs_wave_free(&wave);
		return status ? 1 : 0;
	}
#endif
	/* ASR must bypass the Llama GGUF/checkpoint loader entirely. */
	if (strcmp(mode,"asr") == 0) {
#ifdef CATS_HAYAMIMI
		return cats_hayamimi_run(asr_root,asr_lib,asr_wav,asr_lang,asr_threads,asr_raw,asr_partial);
#else
		fprintf(stderr,"asr mode disabled; rebuild with -DCATS_HAYAMIMI\n");
		return 1;
#endif
	}
	if (checkpoint_path == NULL && model == NULL) {
		fprintf(stderr, "Error: set model with -m file.gguf or checkpoint with -c file.bin\n");
		return 1;
	}
	// parameter validation/overrides
	if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
	if (m->temperature < 0.0) m->temperature = 0.0;
	if (m->topp < 0.0 || 1.0 < m->topp) m->topp = 0.9;
	if (m->steps < 0) m->steps = 0;

	// read weight file
	if (model) {
		if (strstr(model, ".gguf")) cats_gguf_load(model, m); //m->seq_len = 2048;
		else cats_ggml_load(model, m);
		m->max_token_length = 512;
//		m->max_token_length = 4096;
	} else {
		cats_checkpoint_load(checkpoint_path, m);

		// read in the tokenizer.bin file
		m->vocab = (char**)malloc(m->n_vocab * sizeof(char*));
		m->score = (float*)malloc(m->n_vocab * sizeof(float));
		{
			int fd = open(tokenizer_path, O_RDONLY);
			if (fd==-1) {
				fprintf(stderr, "Couldn't open file %s\n", tokenizer_path);
				return 1;
			}
			read(fd, &m->max_token_length, sizeof(uint32_t));
//			printf("max token length: %d\n", m->max_token_length); // 27
			cats_read_vocab(fd, m->n_vocab, m->vocab, m->score, 1);
			close(fd);
		}
	}
	cats_llm_malloc(m);

	printf("transformer dimension: %d\n", m->n_embd);
	printf("ffn layers's dimension: %d\n", m->n_hidden);
	printf("number of layers: %d\n", m->n_layer);
	printf("number of query heads: %d\n", m->n_head);
	printf("number of key/value heads: %d\n", m->n_kv_head);
	printf("vocabulary size: %d\n", m->n_vocab);
	printf("max sequence length: %d\n", m->seq_len);
	printf("seed: %llu\n\n", (unsigned long long)rng_seed);

	// right now we cannot run for more than seq_len steps
	if (m->steps <= 0 || m->steps > m->seq_len) m->steps = m->seq_len;

	// run!
	if (strcmp(mode, "generate") == 0) {
		cats_llm_generate(m, prompt);
	} else if (strcmp(mode, "chat") == 0) {
		cats_llm_chat(m, prompt, system_prompt);
	} else if (strcmp(mode, "kana") == 0) {
		// Convenience: -a kana -p "とうきょう"  → builds <h2k>…<sep>
		char out[512];
		const char *yomi = prompt ? prompt : "";
		m->temperature = 0.0f;
		if (m->steps <= 0 || m->steps > m->seq_len) m->steps = m->seq_len;
#ifdef CATS_HAVE_KANA
		int n = cats_kana2kanji(m, yomi, out, sizeof(out));
		if (n > 0) printf("%s\n", out);
		else fprintf(stderr, "kana2kanji: empty result\n");
#else
		(void)out; (void)yomi;
		fprintf(stderr, "kana mode requires optional catseye_kana.h\n");
#endif
	} else {
		fprintf(stderr, "unknown mode: %s\n", mode);
		error_usage(argv[0]);
	}

	cats_llm_free(m);

	return 0;
}
