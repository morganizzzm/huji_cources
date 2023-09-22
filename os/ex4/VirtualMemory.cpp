#include "VirtualMemory.h"
#include "PhysicalMemory.h"



int getCycDist(uint64_t virtualPageNum, word_t pageSwappedIn, uint64_t i);

uint64_t returnRowInLayer(uint64_t virtualAddress, uint64_t layerNum){
    uint64_t a = (virtualAddress %(1LL << ( (TABLES_DEPTH - layerNum +1 ) * OFFSET_WIDTH )));
    uint64_t b = (1LL << ( (TABLES_DEPTH - layerNum)*OFFSET_WIDTH ));
    return a/b;
}

void cleanFrame(int layer){
    ////sets into given layer
    for (uint64_t i=0; i<PAGE_SIZE; i++){
        PMwrite(layer*PAGE_SIZE+ i, 0);
    }
}

int returnModule(int x){
    if (x<0){
        return -1*x;
    }
    return x;
}

int min(int a, int b){
    if (a<b){
        return a;
    }
    return b;
}




void VMinitialize(){
    cleanFrame(0);

}

void findPageToEvict(uint64_t curFrame, uint64_t depth, word_t* maxCycFrame, int* maxCyclicDist, uint64_t virtualPageNum,
                     word_t pageSwappedIn, word_t* evictedFrameFather, word_t* evictedPageIndex){
    if (depth >= TABLES_DEPTH) {
        return;
    }
    word_t addrNewFrame;
    for (uint64_t i=0; i<  PAGE_SIZE ; i++){
        PMread(PAGE_SIZE * (curFrame) +i, &addrNewFrame);
        if (addrNewFrame != 0){
            if (depth == TABLES_DEPTH - 1) {
                int cycDist = getCycDist(virtualPageNum, pageSwappedIn, i);

                if(cycDist>(*maxCyclicDist)){
                    *maxCyclicDist = cycDist;
                    *maxCycFrame = addrNewFrame;
                    *evictedPageIndex = PAGE_SIZE * virtualPageNum +i;
                    *evictedFrameFather = PAGE_SIZE * (curFrame) + i;
                }
                continue;
            }
            findPageToEvict(addrNewFrame,depth+1, maxCycFrame,maxCyclicDist, virtualPageNum*PAGE_SIZE + i,
                            pageSwappedIn, evictedFrameFather, evictedPageIndex);

        }
    }
}

int getCycDist(uint64_t virtualPageNum, word_t pageSwappedIn, uint64_t i) {
    return min((int)(NUM_PAGES) - returnModule((int)(virtualPageNum * PAGE_SIZE + i - pageSwappedIn)),
               returnModule((int)(virtualPageNum*PAGE_SIZE + i - pageSwappedIn)));
}


void checkIfEvictionNeeded(uint64_t curFrameIdx, uint64_t* frameWithEmptyIdx,  int depth, word_t* maxFrameIndex,
                           bool* isEmptyFrameFound, uint64_t * lastPageWithZeros, uint64_t curFrameDad, uint64_t* frameWithEmptyDad){
    ////finds empty frame in the tree
    if (depth >= TABLES_DEPTH) {
        return;
    }
    word_t newFrameIdx;
    bool isEmpty = true;
    for (uint64_t i=0; i<  PAGE_SIZE ; ++i){
        PMread(PAGE_SIZE * curFrameIdx +i, &newFrameIdx);
        if (newFrameIdx != 0){
            isEmpty = false;
            if (newFrameIdx > *maxFrameIndex ){
                *maxFrameIndex = newFrameIdx;
            }
            checkIfEvictionNeeded(newFrameIdx, frameWithEmptyIdx,  depth+1, maxFrameIndex, isEmptyFrameFound, lastPageWithZeros, PAGE_SIZE * (curFrameIdx) +i, frameWithEmptyDad);
        }
    }
    /// we can enter this if only if empty frame found and this empty frame is not the one created recently
    if (isEmpty && *lastPageWithZeros!= curFrameIdx && !*frameWithEmptyIdx){
        *isEmptyFrameFound = true;
        *frameWithEmptyIdx = curFrameIdx;
        *frameWithEmptyDad = curFrameDad;
    }

}

void traverseTree(uint64_t virtualAddress, uint64_t rowInTable, uint64_t * ptrFather, uint64_t * ptrSon, uint64_t * lastPageWithZeros, int depth){
    uint64_t frameWithEmptyIdx = 0;
    bool isEmptyFrameFound = false;
    word_t maxFrameIdx = 0;
    uint64_t frameWithEmptyDad = 0;

    /// looks for empty frame, if it's found stores its index in frameWithEmptyIdx +  computes the maximal index of the frame in use
    checkIfEvictionNeeded(0, &frameWithEmptyIdx, 0, &maxFrameIdx, &isEmptyFrameFound, lastPageWithZeros, 0, &frameWithEmptyDad);
    if (!isEmptyFrameFound){
        if (maxFrameIdx + 1 < NUM_FRAMES) {
            *ptrSon = maxFrameIdx +1;
        }
        else{
            word_t frameToEvict = 0;
            word_t frameToEvictDad = 0;
            int  maxCyclicDist = 0;
            word_t pageToEvict = 0;

            uint64_t pageSwappedIn = virtualAddress / PAGE_SIZE;
            findPageToEvict(0, 0, &frameToEvict, &maxCyclicDist,  0, pageSwappedIn,
                            &frameToEvictDad, &pageToEvict );
            PMevict(frameToEvict, pageToEvict);
            PMwrite(frameToEvictDad, 0);
            *ptrSon = frameToEvict;
        }

    }
    if (isEmptyFrameFound){
        PMwrite(frameWithEmptyDad, 0);
        *ptrSon = frameWithEmptyIdx;
    }
    PMwrite(PAGE_SIZE* (*ptrFather) + rowInTable, (word_t)*ptrSon );
    if (depth < TABLES_DEPTH -1 ){
        for (int i=0; i<PAGE_SIZE; i++) {
            PMwrite(*ptrSon * PAGE_SIZE + i, 0);
        }
    }

    *lastPageWithZeros = *ptrSon;


}


uint64_t setPage(uint64_t virtualAddress) {
    uint64_t ptrFather = 0;
    uint64_t ptrSon = 0;
    bool flag = true;
    uint64_t lastPageWithZeros = 0;
    for (uint64_t i=0; i<TABLES_DEPTH; i++){
        uint64_t rowInTable = returnRowInLayer(virtualAddress, i);

        PMread( (uint64_t)(ptrSon*PAGE_SIZE+ rowInTable), (word_t*)&ptrFather);
        if (ptrFather == 0) {
            flag = false;
            ptrFather = ptrSon;
            traverseTree(virtualAddress, rowInTable, &ptrFather, &ptrSon, &lastPageWithZeros, i);
        }
        else {
            ptrSon = ptrFather;
            lastPageWithZeros = ptrFather;
        }
    }
    if (!flag){
        PMrestore(ptrSon, virtualAddress / PAGE_SIZE);
    }
    return ptrSon;

}



/* Reads a word from the given virtual address
 * and puts its content in *value.
 *
 * returns 1 on success.
 * returns 0 on failure (if the address cannot be mapped to a physical
 * address for any reason)
 */
int VMread(uint64_t virtualAddress, word_t* value){
    if(virtualAddress >= VIRTUAL_MEMORY_SIZE){
        return 0;
    }
    uint64_t physicalAddress = setPage(virtualAddress) * PAGE_SIZE;
    uint64_t offset = virtualAddress % (1LL << OFFSET_WIDTH);
    PMread(physicalAddress + offset,value);
    return 1;

}

/* Writes a word to the given virtual address.
 *
 * returns 1 on success.
 * returns 0 on failure (if the address cannot be mapped to a physical
 * address for any reason)
 */
int VMwrite(uint64_t virtualAddress, word_t value){
    if(virtualAddress >= VIRTUAL_MEMORY_SIZE){
        return 0;
    }
    uint64_t physicalAddress = setPage(virtualAddress) * PAGE_SIZE;
    uint64_t offset = virtualAddress % (1LL << OFFSET_WIDTH);
    PMwrite(physicalAddress + offset,value);
    return 1;

}

