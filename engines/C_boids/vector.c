#include "vector.h"
#include <math.h>


Vector2 vector_add(Vector2 a, Vector2 b){
    return (Vector2){
        a.x + b.x,
        a.y + b.y
    };
}


Vector2 vector_subtract(Vector2 a, Vector2 b){
    return (Vector2){
        a.x - b.x,
        a.y - b.y
    };
}


Vector2 vector_multiply(Vector2 v, float scalar){
    return (Vector2){
        v.x * scalar,
        v.y * scalar
    };
}


float vector_length(Vector2 v){
    return sqrtf(v.x * v.x + v.y * v.y);
}


Vector2 vector_normalize(Vector2 v){
    float length = vector_length(v);

    if (length == 0)
        return (Vector2){0, 0};

    return (Vector2){
        v.x / length,
        v.y / length
    };
}