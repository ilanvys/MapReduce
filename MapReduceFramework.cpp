#include "MapReduceFramework.h"
#include "Barrier.h"

#include <atomic>
#include <iostream>
#include <pthread.h>
#include <algorithm>

using std::vector;
using std::atomic;
using std::sort;

struct SharedContext {
    const InputVec& inputVec;
    vector<IntermediateVec*>* interVecs;
    vector<IntermediateVec*>* interVecsShuffled;
    OutputVec*  outputVec;
    const MapReduceClient& client;
    atomic<int>* atomic_counter;
    Barrier* barrier;
    pthread_mutex_t* mutex;
    pthread_mutex_t* updateStatusMutex;
    stage_t currentStatus;
    int currentSum;
    atomic<int>* currentProcessed;
    bool shuffle;
};

struct ThreadContext {
    SharedContext* sharedData;
    IntermediateVec* myInterVec;
    int size;
};

struct JobInfo {
    SharedContext* sharedContext;
    pthread_t* threads;
    ThreadContext* contexts;
    int numThreads;
    pthread_mutex_t* waitMutex;
    bool waitCalled;
};

void mutexLock(pthread_mutex_t* mutex) {
    if (pthread_mutex_lock(mutex) != 0) {
        std::cout <<  "system error: error on pthread_mutex_lock" << std::endl;
        exit(1);
    }
}

void mutexUnlock(pthread_mutex_t* mutex) {
    if (pthread_mutex_unlock(mutex) != 0) {
        std::cout <<  "system error: error on pthread_mutex_unlock" << std::endl;
        exit(1);
    }
}

void mutexDestroy(pthread_mutex_t* mutex) {
    if (pthread_mutex_destroy(mutex) != 0) {
        std::cout <<  "system error: error on pthread_mutex_destroy" << std::endl;
        exit(1);
    }
}

static bool sortByK2(const IntermediatePair &a, const IntermediatePair &b) {
    return (*(a.first) < *(b.first));
}

void emit2 (K2* key, V2* value, void* context) {
    ThreadContext* threadContext = (ThreadContext*)(context);
    threadContext->myInterVec->push_back({key,value});
}

void emit3 (K3* key, V3* value, void* context) {
    ThreadContext* threadContext = (ThreadContext*)(context);
    mutexLock(threadContext->sharedData->mutex);
    threadContext->sharedData->outputVec->push_back({key,value});
    mutexUnlock(threadContext->sharedData->mutex);
}

void shuffleInterVecs(ThreadContext* context) {
    K2* largestKey;
    while (context->sharedData->currentProcessed->load() < context->sharedData->currentSum) {
        // find the largest key remaining
        largestKey = nullptr;
        for (IntermediateVec* interVec: *context->sharedData->interVecs) {
            if (!interVec->empty()) {
                if (largestKey == nullptr || *largestKey < *interVec->back().first) {
                    largestKey = interVec->back().first;
                }
            }
        }

        // push all pairs with keys equal to 'largestKey' to a single vector
        IntermediateVec* shuffledVec = new IntermediateVec(0);
        for (IntermediateVec* interVec: *context->sharedData->interVecs) {
            while (!interVec->empty() &&
            (!(*interVec->back().first < *largestKey) && !((*largestKey) < *interVec->back().first))) {
                shuffledVec->push_back(interVec->back());
                (*context->sharedData->currentProcessed)++;
                interVec->pop_back();
            }
        }
        context->sharedData->interVecsShuffled->push_back(shuffledVec);
    }
}

void shufflePhase(ThreadContext* context) {
    atomic<int>* counter = context->sharedData->atomic_counter;

    // Update Status -> SHUFFLE_STAGE
    mutexLock(context->sharedData->updateStatusMutex);
    (*context->sharedData->currentProcessed) = 0;
    context->sharedData->currentStatus = SHUFFLE_STAGE;
    context->sharedData->currentSum = 0;
    for (IntermediateVec* interVec: *context->sharedData->interVecs) {
        context->sharedData->currentSum += interVec->size();
    }
    mutexUnlock(context->sharedData->updateStatusMutex);

    // Shuffle
    shuffleInterVecs(context);
    context->sharedData->shuffle = true;

    // Update Status -> REDUCE_STAGE
    mutexLock(context->sharedData->updateStatusMutex);
    (*counter) = 0;
    (*context->sharedData->currentProcessed) = 0;
    context->sharedData->currentStatus = REDUCE_STAGE;
    mutexUnlock(context->sharedData->updateStatusMutex);
}

void* worker(void* arg) {
    ThreadContext* context = (ThreadContext*) arg;
    atomic<int>* counter = context->sharedData->atomic_counter;

    // Update Status -> MAP_STAGE
    mutexLock(context->sharedData->updateStatusMutex);
    if (context->sharedData->currentStatus == UNDEFINED_STAGE) {
        context->sharedData->currentStatus = MAP_STAGE;
        context->sharedData->currentSum = context->sharedData->inputVec.size();
    }
    mutexUnlock(context->sharedData->updateStatusMutex);

    // map phase
    int old_value = (*counter)++;
    while (old_value < context->size) {
        context->sharedData->client.map(context->sharedData->inputVec.at(old_value).first,
                                        context->sharedData->inputVec.at(old_value).second,
                                        context);
        (*context->sharedData->currentProcessed)++;
        old_value = (*counter)++;
    }

    // sort phase
    sort(context->myInterVec->begin(), context->myInterVec->end(), sortByK2);

    mutexLock(context->sharedData->mutex);
    context->sharedData->interVecs->push_back(context->myInterVec);
    mutexUnlock(context->sharedData->mutex);

    // barrier
    context->sharedData->barrier->barrier();

    // shuffle
    mutexLock(context->sharedData->mutex);
    if(!context->sharedData->shuffle){
        shufflePhase(context);
    }
    mutexUnlock(context->sharedData->mutex);

    // reduce
    old_value = (*counter)++;
    while (old_value < (int)context->sharedData->interVecsShuffled->size()) {
        int currentSize =(int) context->sharedData->interVecsShuffled->at
            (old_value)->size();
        context->sharedData->client.reduce(context->sharedData->interVecsShuffled->at(old_value), context);
        context->sharedData->currentProcessed->fetch_add(currentSize);
        old_value = (*counter)++;
    }

    return nullptr;
}

JobHandle startMapReduceJob(const MapReduceClient& client,
	const InputVec& inputVec, OutputVec& outputVec, int multiThreadLevel) {
    vector<IntermediateVec*>* interVecs = new vector<IntermediateVec*>(0);
    vector<IntermediateVec*>* interVecsShuffled = new vector<IntermediateVec*>(0);

    atomic<int>* atomic_counter = new atomic<int>(0);
    atomic<int>* currentProcessed = new atomic<int>(0);

    Barrier* barrier = new Barrier(multiThreadLevel);

    pthread_mutex_t* mutex = new pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);\
    pthread_mutex_t* updateStatusMutex = new pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);

    SharedContext* sharedContext = new SharedContext{inputVec,
                                                     interVecs,
                                                     interVecsShuffled,
                                                     &outputVec,
                                                     client,
                                                     atomic_counter,
                                                     barrier,
                                                     mutex,
                                                     updateStatusMutex,
                                                     UNDEFINED_STAGE,
                                                     0,
                                                     currentProcessed,
                                                     false,
                                                     };

    pthread_t* threads = new pthread_t[multiThreadLevel];
    ThreadContext* contexts = new ThreadContext[multiThreadLevel];
    pthread_mutex_t* waitMutex = new pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
    JobHandle jobHandle =  new JobInfo{sharedContext,
                                       threads,
                                       contexts,
                                       multiThreadLevel,
                                       waitMutex,
                                       false};

    // create threads
    for (int i = 0; i < multiThreadLevel; ++i) {
        IntermediateVec* intermediateVec = new IntermediateVec(0);
        contexts[i]= {sharedContext, intermediateVec,(int) inputVec.size()};
        if(pthread_create(threads + i, nullptr, worker, contexts + i) != 0){
            std::cout <<  "system error: error on pthread_create\n" << std::endl;
            exit(1);
        }
    }

    return jobHandle;
}

void waitForJob(JobHandle job) {
    JobInfo* jobInfo = (JobInfo*) job;
    mutexLock(jobInfo->waitMutex);
    if(!(jobInfo->waitCalled)) {
        jobInfo->waitCalled = true;
        for (int i = 0; i < jobInfo->numThreads; ++i) {
            pthread_join(jobInfo->threads[i], nullptr);
        }
    }
    mutexUnlock(jobInfo->waitMutex);
}

void getJobState(JobHandle job, JobState* state) {
    JobInfo* jobInfo = (JobInfo*) job;
    mutexLock(jobInfo->sharedContext->updateStatusMutex);
    state->stage = jobInfo->sharedContext->currentStatus;
    if (jobInfo->sharedContext->currentSum == 0) {
        state->percentage = 0;
    }
    else if ((jobInfo->sharedContext->currentProcessed->load()) > jobInfo->sharedContext->currentSum) {
            state->percentage = 100;
    }
    else {
        state->percentage = (float ) ((( jobInfo->sharedContext->currentProcessed->load()) * 100) / jobInfo->sharedContext->currentSum);
    }
    mutexUnlock(jobInfo->sharedContext->updateStatusMutex);
}

void closeJobHandle(JobHandle job) {
    waitForJob(job);
    JobInfo* jobInfo = (JobInfo*) job;

    for (int i = 0; i <(int) jobInfo->sharedContext->interVecs->size(); ++i) {
        delete jobInfo->sharedContext->interVecs->at(i);
    }
    delete jobInfo->sharedContext->interVecs;

    for (int i = 0; i < (int)jobInfo->sharedContext->interVecsShuffled->size(); ++i) {

        delete jobInfo->sharedContext->interVecsShuffled->at(i);
    }
    delete jobInfo->sharedContext->interVecsShuffled;
    delete jobInfo->sharedContext->atomic_counter;
    delete jobInfo->sharedContext->currentProcessed;
    delete jobInfo->sharedContext->barrier;

    mutexDestroy(jobInfo->sharedContext->mutex);
    delete jobInfo->sharedContext->mutex;

    mutexDestroy(jobInfo->sharedContext->updateStatusMutex);
    delete jobInfo->sharedContext->updateStatusMutex;

    delete jobInfo->sharedContext;

    mutexDestroy(jobInfo->waitMutex);
    delete jobInfo->waitMutex;

    delete[] jobInfo->threads;
    delete[] jobInfo->contexts;
    delete jobInfo;
}
