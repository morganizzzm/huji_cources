
#include <atomic>
#include <cstdlib>
#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include "Barrier.h"
#include <algorithm>
#include <iostream>
#include <semaphore.h>
#include <valarray>

#define SYS_ERROR_MUTEX "system error: error on pthread_mutex_lock\n"
#define SYS_ERROR_THREAD_CREATE "system error: error thread creation\n"
#define SYS_ERROR_INITIALIZATION "system error: initialization fail\n"


typedef void* JobHandle;


struct ThreadContext;

struct JobContext{
    int multiThreadLevel;
    std::atomic<int>  counterInSorted ;
    int sortedToShuffleSize = 0;
    int shuffleSize  =0;
    int isDone = 0;
    std::atomic<uint64_t> atomic_counter;
    Barrier* barrier;
    std::vector<IntermediateVec * >* toShuffle;
    std::vector<IntermediateVec * >* sortedToShuffle;
    pthread_t* threads;
    const MapReduceClient *client ;
    const InputVec* inputVec;
    OutputVec* outputVec;
    ThreadContext* contexts;
    pthread_mutex_t ADD_TO_SHUFFLE;
    pthread_mutex_t STAGE_PROTECTOR;
    pthread_mutex_t REDUCE;
    pthread_mutex_t SWITCH_TO_MAP;
    pthread_mutex_t SHUFFLE_SIZE_LOCK;
};

struct ThreadContext {
    int threadID ;
    std::vector<std::pair<K2*, V2*>>* curK2V2;
    JobContext* jobContext;


};

void safe_mutex_lock(pthread_mutex_t* mutex){
    /// safe mutex lock
    if (pthread_mutex_lock(mutex) != 0){
        fprintf(stdout, SYS_ERROR_MUTEX);
        exit(1);
    }
}

void safe_mutex_unlock(pthread_mutex_t* mutex){
    ///safe mutex unlock
    if (pthread_mutex_unlock(mutex) != 0){
        fprintf(stdout, SYS_ERROR_MUTEX);
        exit(1);
    }
}

void threadShuffle(ThreadContext* jobContext);


void emit2 (K2* key, V2* value, void* context){
    ThreadContext* threadContext = static_cast<ThreadContext*>(context);
    IntermediatePair newPair(key, value);
    //// DOESN'T NEED MUTEX SINCE SINCE EVERY THREAFD HAS ITS OWN CURK2V2
    threadContext->curK2V2->push_back(newPair);
    safe_mutex_lock(&(threadContext->jobContext->SHUFFLE_SIZE_LOCK));
    /// START OF CRITICAL SECTION: shuffleSize is mutual for all the threads
    threadContext->jobContext->shuffleSize++;
    /// END OF CRITICAL SECTION
    safe_mutex_unlock(&(threadContext->jobContext->SHUFFLE_SIZE_LOCK));

}

void emit3 (K3* key, V3* value, void* context){
    OutputPair newPair(key, value);
    ThreadContext* threadContext = (ThreadContext*)(context);
    safe_mutex_lock(&threadContext->jobContext->REDUCE);
    /// START OF CRITICAL SECTION: outputVec is mutual for all the threads
    threadContext->jobContext->outputVec->push_back(newPair);
    /// END OF CRITICAL SECTION
    safe_mutex_unlock(&threadContext->jobContext->REDUCE);

}

bool comparatorK2V2(const std::pair<K2*, V2*>& pair1, const std::pair<K2*, V2*>& pair2){
    return (*(pair1.first) < *(pair2.first));
}

bool comparatorVectorVector( std::vector<std::pair<K2*, V2*>> * vec1,  std::vector<std::pair<K2*, V2*>>* vec2){
    return (*(vec1->back().first) < *(vec2->back().first));
}


void switchToShuffle( JobContext *jobContext);

void switchToReduce( JobContext *jobContext);

void mapAndSort(ThreadContext *jobContext);

void reduce(ThreadContext *threadContext );

JobContext *initializeJobContext(int multiThreadLevel, const MapReduceClient &client, const InputVec &inputVec,
                                OutputVec &outputVec);

bool K2V2equal(IntermediatePair pair1, IntermediatePair pair2){
    return !(*(pair1.first) < *(pair2.first)) && !((*pair2.first) < (*pair1.first));
}




void* mappingThreads(void* arg){
    // The func performs mapping and sorting of single pair <K1,V1>
    // Can be called by number of threads in parallel
    ThreadContext* threadContext = static_cast<ThreadContext*>(arg);
    JobContext* jobContext = threadContext->jobContext;
    ////// mapping and sorting stage
    mapAndSort(threadContext);
    ///// only after all the threads finished the job can start shuffling
    jobContext->barrier->barrier();
    ///// shuffle stage
    threadShuffle(threadContext);
    //// wait till shuffling is done
    jobContext->barrier->barrier();
    //// reduce stage
    reduce(threadContext);
    return nullptr;
}

void reduce(ThreadContext *threadContext) {
    while(true){

        int idx = threadContext->jobContext->counterInSorted.fetch_add(1);
        //////IF ALL THE KEYS WERE PROCESSED RETURN MUTEX
        if (idx>= threadContext->jobContext->sortedToShuffleSize){
            break;
        }
        ////GET CORRESPONDING INTERMEDIATE VECTOR
        IntermediateVec * vec = (*threadContext->jobContext->sortedToShuffle)[idx];
        threadContext->jobContext->client->reduce(vec, threadContext);
        threadContext->jobContext->atomic_counter.fetch_add((*vec).size());

    }

}

void mapAndSort(ThreadContext *threadContext) {
    safe_mutex_lock(&threadContext->jobContext->SWITCH_TO_MAP);
    ///// SET ATOMIC COUNTER"S 2 LAST BITS TO 1 AND MIDDLE 31 BITS TO NUMBER OF <K1,V1>
    if (threadContext->jobContext->atomic_counter.load() >> 62 == 0){
        unsigned long num = ( (long)1 << 62) + (threadContext->jobContext->inputVec->size() <<31);
        threadContext->jobContext->atomic_counter.exchange(num);
    }
    safe_mutex_unlock(&threadContext->jobContext->SWITCH_TO_MAP);
    ///// MAP STAGE
    JobContext* jobContext = threadContext->jobContext;
    while (true){
        unsigned long oldAtomic = threadContext->jobContext->atomic_counter.fetch_add(1);
        int curKey = (int)((oldAtomic<< 33) >>33);
        int allKeys = (int)((oldAtomic << 2) >> 33);
        if ( curKey>= allKeys ) {
            break;
        }
        const K1* k1 = (*jobContext->inputVec)[curKey].first;
        const V1* v1 = (*jobContext->inputVec)[curKey].second;
        threadContext->jobContext->client->map(k1, v1,
                                threadContext);

    }
    //// SORT STAGE
    std::sort(threadContext->curK2V2->begin(), threadContext->curK2V2->end(), comparatorK2V2);
    safe_mutex_lock(&threadContext->jobContext->ADD_TO_SHUFFLE);
    /// ADD SORTED INTERMEDIATE VECTOR TO TO_SHUFFLE VECTOR THAT WILL BE USED AT SHUFFLING STAGE
    /// START OF CRITICAL SECTION :toShuffle is mutual for all threads
    threadContext->jobContext->toShuffle->push_back(threadContext->curK2V2);
    /// END OF CRITICAL SECTION
    safe_mutex_unlock(&threadContext->jobContext->ADD_TO_SHUFFLE);
}





void threadShuffle(ThreadContext* threadContext) {

    //We have a vector of vectors of pairs so that each vector is an element of initial input vector split into pairs
    // of <K2, V2>.  We sort the vectors respective to tke key values of pair, and check the equality of the biggest
    // elements before inserting them in the vector of the same keys
//    pthread_mutex_lock(&LET_ME_SHUFFLE);
    if (threadContext->threadID != 0){
        return;
    }
    switchToShuffle(threadContext->jobContext);
    auto it = std::remove_if(threadContext->jobContext->toShuffle->begin(), threadContext->jobContext->toShuffle->end(),
                             [](std::vector<std::pair<K2*, V2*>>* vec) {
                                 return vec->empty();
                             });
    threadContext->jobContext->toShuffle->erase(it, threadContext->jobContext->toShuffle->end());

    while(true){
        if (threadContext->jobContext->toShuffle->begin() == threadContext->jobContext->toShuffle->end()){
            break;
        }


        std::sort(threadContext->jobContext->toShuffle->begin(), threadContext->jobContext->toShuffle->end(),
                  comparatorVectorVector);
        if (threadContext->jobContext->toShuffle->back()->empty()){  ///toShuffle consists only of empty vectors
            break;

        }
        const IntermediatePair K2ToPop = (threadContext->jobContext->toShuffle->back()->back());
        IntermediateVec * vecEqualK2 = new IntermediateVec ();
        std::vector<IntermediateVec * >::reverse_iterator it = threadContext->jobContext->toShuffle->rbegin();
        while ( it!= threadContext->jobContext->toShuffle->rend()){

            if (!K2V2equal(K2ToPop, (*it)->back())){
                break;
            }
            while ((K2V2equal(K2ToPop, (*it)->back()))){
                vecEqualK2->push_back((*it)->back());
                (*it)->pop_back();
                threadContext->jobContext->atomic_counter++;
                if ((*it)->empty()){
                    break;
                }
            }

            if ((*it)->empty()){

                it = decltype(it)(threadContext->jobContext->toShuffle->erase(std::next(it).base()));
                if(it==threadContext->jobContext->toShuffle->rend() || *it == nullptr){
                    break;
                }
                continue;

            }
            ++it;
        }
        threadContext->jobContext->sortedToShuffle->push_back(vecEqualK2);
        threadContext->jobContext->sortedToShuffleSize ++; /// NOT a critical section
    }
    switchToReduce(threadContext->jobContext);

}

void switchToReduce( JobContext *jobContext) {
    unsigned long num = (((long)3 << 62) + ((long) jobContext->shuffleSize << 31));
    jobContext->atomic_counter.exchange(num);
}

void switchToShuffle( JobContext *jobContext) {
    long num =  ( ((long)2 << 62) +  ( (long) jobContext->shuffleSize << 31));
    jobContext->atomic_counter.exchange(num);

}

JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel){
   JobContext* jobContext= initializeJobContext(multiThreadLevel, client, inputVec, outputVec);

    for (int i = 0; i < multiThreadLevel; ++i) {
        jobContext->contexts[i].threadID = i;
        jobContext->contexts[i].jobContext = jobContext;
        jobContext->contexts[i].curK2V2 = new IntermediateVec ();
        if (pthread_create(jobContext->threads +i, nullptr, mappingThreads, &(jobContext->contexts[i]))){
            fprintf(stdout, SYS_ERROR_THREAD_CREATE);
            exit(1);

        }
    }

    return jobContext;

}

JobContext *initializeJobContext(int multiThreadLevel, const MapReduceClient &client, const InputVec &inputVec,
                                OutputVec &outputVec) {
    /// This function initializes JobCpntext used to monitor the state of the current job
    auto * jobContext = new(std::nothrow) JobContext;
    if(!jobContext){
        fprintf(stdout, SYS_ERROR_INITIALIZATION);
        exit(1);
    }
    jobContext->threads = new pthread_t[multiThreadLevel];
    if(!jobContext->threads){
        fprintf(stdout, SYS_ERROR_INITIALIZATION);
        exit(1);
    }
    jobContext->inputVec = &inputVec;
    jobContext->outputVec = &outputVec;
    jobContext->atomic_counter = 0;
    jobContext->counterInSorted =0;
    jobContext->barrier = new Barrier(multiThreadLevel);
    if(!jobContext->barrier){
        fprintf(stdout, SYS_ERROR_INITIALIZATION);
        exit(1);
    }
    jobContext->contexts = new ThreadContext[multiThreadLevel];
    if(!jobContext->contexts){
        fprintf(stdout, SYS_ERROR_INITIALIZATION);
        exit(1);
    }
    jobContext->client = &client;
    jobContext->multiThreadLevel = multiThreadLevel;
    jobContext->toShuffle= new std::vector<IntermediateVec*>();
    jobContext->sortedToShuffle = new std::vector<IntermediateVec*>();
    jobContext->ADD_TO_SHUFFLE = PTHREAD_MUTEX_INITIALIZER;
    jobContext-> STAGE_PROTECTOR = PTHREAD_MUTEX_INITIALIZER;
    jobContext-> REDUCE = PTHREAD_MUTEX_INITIALIZER;
    jobContext-> SWITCH_TO_MAP = PTHREAD_MUTEX_INITIALIZER;
    jobContext-> SHUFFLE_SIZE_LOCK =PTHREAD_MUTEX_INITIALIZER;
    return jobContext;
}

void waitForJob(JobHandle job){
    JobContext* jobContext = static_cast<JobContext*>(job);
    ////in case this function is called twice
    if (jobContext->isDone ){
        return;
    }
    for (int i=0; i<jobContext->multiThreadLevel; i++){
        pthread_join( (jobContext->threads[i]), NULL);

    }
    jobContext->isDone = 1;
}



void getJobState(JobHandle job, JobState* state){
    /// FINE!
    JobContext* jobToDo = (JobContext*)job;
    safe_mutex_lock(&jobToDo->STAGE_PROTECTOR);
    ///START OF CRITICAL SECTION
    unsigned long oldAtomic = jobToDo->atomic_counter.load();
    state->stage = static_cast<stage_t>(oldAtomic>>62);
    state->percentage = (float)static_cast<float>((oldAtomic << 33) >>33 ) /
            static_cast<float>((oldAtomic << 2) >> 33 )*100;
    if (state->percentage > 100){ //in case the number of percentage is bigger than 100
        state->percentage = 100;
    }
    //// END OF CRITICAL SECTION
    safe_mutex_unlock(&jobToDo->STAGE_PROTECTOR);

}


void safe_mutex_destroy(pthread_mutex_t* mutex){
    if (pthread_mutex_destroy(mutex) != 0) {
		fprintf(stdout, "[[Barrier]] error on pthread_mutex_destroy");
		exit(1);
	}
}

void freeMemory(JobHandle job){
    JobContext* jobContext = (JobContext*)job;
    safe_mutex_destroy(&jobContext->ADD_TO_SHUFFLE);
    safe_mutex_destroy(&jobContext->STAGE_PROTECTOR);
    safe_mutex_destroy(&jobContext->REDUCE);
    safe_mutex_destroy(&jobContext->SWITCH_TO_MAP);
    safe_mutex_destroy(&jobContext->SHUFFLE_SIZE_LOCK);
    for (int i=0; i<jobContext->multiThreadLevel; i++){
        delete jobContext->contexts[i].curK2V2;
    }
    delete jobContext->barrier;
    delete jobContext->toShuffle;
    for (int i=0; i<jobContext->sortedToShuffleSize; i++){
        delete (*jobContext->sortedToShuffle)[i];
    }
    delete jobContext->sortedToShuffle;
    delete []jobContext->contexts;
    delete []jobContext->threads;
    delete jobContext;

}

void closeJobHandle(JobHandle job){
    waitForJob(job);
    freeMemory(job);
}


