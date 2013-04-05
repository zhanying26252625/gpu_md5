#include <string.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <valarray>

#include <cuda_runtime_api.h>

#include <helper_cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>
#include <helper_string.h>
#include <helper_timer.h>

using namespace std;

int niters = 10;

// Some declarations that should wind up in their own .h file at some point
void print_md5(uint *hash, bool crlf = true);
void md5_prep(char *c0);
double execute_kernel(int blocks_x, int blocks_y, int threads_per_block, int shared_mem_required, int realthreads, uint *gpuWords, uint *gpuHashes, bool search = false);
void init_constants(uint *target = NULL);
void md5_cpu(uint w[16], uint &a, uint &b, uint &c, uint &d);
void md5_cpu_v2(const uint *in, uint &a, uint &b, uint &c, uint &d);
int deviceQuery();

///////////////////////////////////////////////////////////
// CUDA helpers

//
// Find the dimensions (bx,by) of a 2D grid of blocks that 
// has as close to nblocks blocks as possible
//
void find_best_factorization(int &bx, int &by, int nblocks)
{
	bx = -1;
	int best_r = 100000;
	for(int bytmp = 1; bytmp != 65536; bytmp++)
	{
		int r  = nblocks % bytmp;
		if(r < best_r && nblocks / bytmp < 65535)
		{
			by = bytmp;
			bx = nblocks / bytmp;
			best_r = r;
			
			if(r == 0) { break; }
			bx++;
		}
	}
	if(bx == -1) { std::cerr << "Unfactorizable?!\n"; exit(-1); }
}

//
// Given a total number of threads, their memory requirements, and the
// number of threadsPerBlock, compute the optimal allowable grid dimensions.
// Returns false if the requested number of threads are impossible to fit to
// shared memory.
//
bool calculate_grid_parameters(int gridDim[3], int threadsPerBlock, int neededthreads, int dynShmemPerThread, int staticShmemPerBlock)
{
	const int shmemPerMP =  16384;

	int dyn_shared_mem_required = dynShmemPerThread*threadsPerBlock;
	int shared_mem_required = staticShmemPerBlock + dyn_shared_mem_required;
	if(shared_mem_required > shmemPerMP) { return false; }

	// calculate the total number of threads
	int nthreads = neededthreads;
	int over = neededthreads % threadsPerBlock;
	if(over) { nthreads += threadsPerBlock - over; } // round up to multiple of threadsPerBlock

	// calculate the number of blocks
	int nblocks = nthreads / threadsPerBlock;
	if(nthreads % threadsPerBlock) { nblocks++; }

	// calculate block dimensions so that there are as close to nblocks blocks as possible
	find_best_factorization(gridDim[0], gridDim[1], nblocks);
	gridDim[2] = 1;

	return true;
}

//
// Shared aux. functions (used both by GPU and CPU setup code)
//

union md5hash
{
	uint ui[4];
	char ch[16];
};

/*

static const char HEX[16] = {

'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'

};

void printMD5hash(md5hash hash){

	std::string str;

	for(int i =0 ;i < 16 ;i ++){

		unsigned int t = hash.ch[i];
		
		unsigned int a = t/16;

		unsigned int b = t%16;

		str.append(1,HEX[a]);

		str.append(1,HEX[b]);
	}

	cout<<endl<<str<<endl;
}

*/

//
// Convert an array of null-terminated strings to an array of 64-byte words
// with proper MD5 padding
//
void md5_prep_array(std::valarray<char> &paddedWords, const std::vector<std::string> &words)
{
	paddedWords.resize(64*words.size());
	paddedWords = 0;

	for(int i=0; i != words.size(); i++)
	{
		char *w = &paddedWords[i*64];
		strncpy(w, words[i].c_str(), 56);
		md5_prep(w);
	}
}


//
// GPU calculation: given a vector ptext of plain text words, compute and
// return their MD5 hashes
//
int cuda_compute_md5s(std::vector<md5hash> &hashes, const std::vector<std::string> &ptext, uint *target = NULL, bool benchmark = false)
{
	//CUT_DEVICE_INIT();

	// load the MD5 constant arrays into GPU constant memory
	init_constants(target);


	// pad dictionary words to 64 bytes (MD5 block size)
	std::valarray<char> paddedWords;
	md5_prep_array(paddedWords, ptext);

	// Upload the dictionary onto the GPU device
	uint *gpuWords, *gpuHashes = NULL;
	( cudaMalloc((void **)&gpuWords, paddedWords.size()) );
	( cudaMemcpy(gpuWords, &paddedWords[0], paddedWords.size(), cudaMemcpyHostToDevice) );

	if(target != NULL)
	{
		// allocate GPU memory for match signal, instead of the actual hashes
		( cudaMalloc((void **)&gpuHashes, 4*sizeof(uint)) );
		uint tmp[4] = {0}; // initialize to zero
		( cudaMemcpy(gpuHashes, tmp, 4*sizeof(uint), cudaMemcpyHostToDevice) );
	}
	else
	{
		// allocate GPU memory for computed hashes
		( cudaMalloc((void **)&gpuHashes, 4*sizeof(uint)*ptext.size()) );
	}

	//
	// The tricky part: compute the optimal number of threads per block,
	// and the number of blocks
	//
	int dynShmemPerThread = 64;	// built in the algorithm
	int staticShmemPerBlock = 32;	// read from .cubin file

	double bestTime = 1e10, bestRate = 0.;
	int bestThreadsPerBlock;
	int nthreads = ptext.size();
	int tpb = benchmark ? 10 : 63;	// tpb is number of threads per block
					// 63 is the experimentally determined best case scenario on my 8800 GTX Ultra
	do
	{
		int gridDim[3];
		if(!calculate_grid_parameters(gridDim, tpb, nthreads, dynShmemPerThread, staticShmemPerBlock)) { continue; }

		// Call the kernel 10 times and calculate the average running time
		double gpuTime = 0.; int k;
		for(k=0; k != niters; k++)
		{
			gpuTime += execute_kernel(gridDim[0], gridDim[1], tpb, tpb*dynShmemPerThread, ptext.size(), gpuWords, gpuHashes, target != NULL);
		}
		gpuTime /= k;
		double rate = 1000 * ptext.size() / gpuTime;

		if(bestRate < rate)
		{
			bestTime = gpuTime;
			bestRate = rate;
			bestThreadsPerBlock = tpb;
		}

		if(benchmark)
		{
			std::cout << "words=" << ptext.size()
				<< " tpb=" << tpb
				<< " nthreads=" << gridDim[0]*gridDim[1]*tpb << " nblocks=" << gridDim[0]*gridDim[1]
				<< " gridDim[0]=" << gridDim[0] << " gridDim[1]=" << gridDim[1]
				<< " padding=" << gridDim[0]*gridDim[1]*tpb - ptext.size()
				<< " dynshmem=" << dynShmemPerThread*tpb
				<< " shmem=" << staticShmemPerBlock + dynShmemPerThread*tpb
				<< " gpuTime=" << gpuTime
				<< " rate=" << (int)rint(rate)
				<< std::endl;
		}

	} while(benchmark && tpb++ <= 512);

	if(benchmark)
	{
		std::cerr << "\nBest case: threadsPerBlock=" << bestThreadsPerBlock << "\n";
	}
	std::cerr << "GPU MD5 time : " <<  bestTime << " ms (" << std::fixed << 1000. * ptext.size() / bestTime << " hash/second)\n";

	// Download the results
	if(target != NULL)
	{
		uint ret[4];
		( cudaMemcpy(ret, gpuHashes, sizeof(uint)*4, cudaMemcpyDeviceToHost) );
		return ret[3] ? ret[0] : -1;
	}
	else
	{
		// Download the computed hashes
		hashes.resize(ptext.size());
		( cudaMemcpy(&hashes.front(), gpuHashes, sizeof(uint)*4*ptext.size(), cudaMemcpyDeviceToHost) );
	}

	// Shutdown
	( cudaFree(gpuWords) );
	( cudaFree(gpuHashes) );

	return 0;
}

int gpuCrack(int argc, char **argv)
{

	//bool devQuery = false, benchmark = false;

	bool hasPass = false;

	// Load plaintext dictionary
	std::vector<std::string> ptext;
	
	if( argc >= 2 ){

		hasPass = true;

		for(int i = 1; i< argc ;i++)
			ptext.push_back( std::string(argv[i]) );
	}
	
	bool devQuery = true, benchmark = false;

	if(devQuery)
		 deviceQuery(); 

	if( !hasPass ){

		std::cerr << "Loading words from stdin ...\n";

		std::string word;

		while(std::cin >> word)
		{
			ptext.push_back(word);
		}

		std::cerr << "Loaded " << ptext.size() << " words.\n\n";
	}
	
	// Do search/calculation
	std::vector<md5hash> hashes_gpu;
	
	// Compute hashes
	cuda_compute_md5s(hashes_gpu, ptext, NULL, benchmark);

	for(int i = 0; i< hashes_gpu.size();i++){

		md5hash hash = hashes_gpu[i];

		cout<<endl<<"MD5 of "<<ptext[i]<<" is :";

		print_md5((uint*)&hash,true);

		//printMD5hash(hash);
	}

	cout<<endl;

	return 0;
}
