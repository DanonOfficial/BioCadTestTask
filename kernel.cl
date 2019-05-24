

float distance(float x1, float y1, float z1, float x2, float y2, float z2){
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
}


// Here is some strange solutions, that search are easily working with recursion or stack, however that very difficult for GPU, moreover we want to have depth no more than 3
// So that's highly specific version of BFS, that looks more like iterative depth version.
unsigned int findDepth(const __global unsigned int *bonds, unsigned int bondsSize, const __global unsigned int *index, unsigned int indexSize, unsigned int  in, unsigned int  key) {

    for(unsigned int i = index[in]; i < (in == indexSize? bondsSize: index[in+1]); i++){ // 1
        if (bonds[i] == key) {
            return 1;
        }
    }
    for(unsigned int i = index[in]; i < (in == indexSize? bondsSize: index[in+1]); i++){ // 2
        for(unsigned int j = index[bonds[i]]; j < (in == indexSize? bondsSize: index[bonds[i]+1]); j++){
            if (bonds[j] == key) {
                return 2;
            }
        }
    }
    for(unsigned int i = index[in]; i < (in == indexSize? bondsSize: index[in+1]); i++){ // 2
        for(unsigned int j = index[bonds[i]]; j < (in == indexSize? bondsSize: index[bonds[i]+1]); j++){
            for(unsigned int k = index[bonds[j]]; k< (in == indexSize? bondsSize: index[bonds[j]+1]); k++){
                if (bonds[k] == key) {
                    return 3;
                }
            }
        }
    }
    return 4;
}


__kernel void solve( const __global float *charges,
                     const __global unsigned int *bonds,
                     const __global unsigned int *indexing,
                     const __global float *atoms,
                     __global float *result,
                     unsigned int amountOfAtoms,
                     unsigned int amountOfBonds,
                     unsigned int n) { //n - sizeof image
    const unsigned int index = get_global_id(0);
    if (index < n) {
        unsigned int i = index / amountOfAtoms;
        unsigned int j = index % amountOfAtoms;
        if(i < j){

            unsigned int depth = findDepth(bonds, amountOfBonds, indexing, amountOfAtoms, i, j);
            float f;
            if (depth < 3) {
                f = 0;
            } else {
                if (depth == 3) {
                    f = 0.5f;
                } else {
                    f = 1.f;
                }
            }
            float dist = distance(atoms[j * 3], atoms[j * 3 + 1], atoms[j * 3 + 2], atoms[i * 3], atoms[i * 3 + 1],
                                   atoms[i * 3 + 2]);
            unsigned int resIndex = (2*(amountOfBonds -1)- (i-1))*i/2 + j -(i+1);
            result[resIndex] =  f * charges[i] * charges[j] / dist;
        }
    }
}
