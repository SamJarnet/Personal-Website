#ifndef BOID_H
#define BOID_H

#define MAX_BOIDS 200
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "vector.h"

typedef struct {
    int members[MAX_BOIDS];
    int count;
} Group;

typedef struct {
    Vector2 position;
    Vector2 velocity;
    bool is_grouped;
} Boid;

void boid_random_init(Boid *boid, int width, int height);
void boid_update(Boid *boid, Vector2 change, float dt);
void check_boundaries(Boid *boid, int width, int height);

int find_groups(Boid *boids, int boid_count, float group_radius, Group *groups_out);

Vector2 separation(Boid *boids, Group *group, int member_index, float separation_strength);
Vector2 cohesion(Boid *boids, Group *group, int member_index, Vector2 avg_pos, float cohesion_strength);
Vector2 alignment(Boid *boids, Group *group, int member_index, Vector2 avg_vel, float alignment_strength);

Vector2 find_average_pos(Boid *boids, Group *group);
Vector2 find_average_vel(Boid *boids, Group *group);

#endif