#ifndef VECTOR_H
#define VECTOR_H

typedef struct{
    float x;
    float y;
} Vector2;


Vector2 vector_add(Vector2 a, Vector2 b);
Vector2 vector_subtract(Vector2 a, Vector2 b);
Vector2 vector_multiply(Vector2 v, float scalar);
float vector_length(Vector2 v);
Vector2 vector_normalize(Vector2 v);

#endif