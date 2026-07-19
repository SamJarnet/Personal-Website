#include <emscripten.h>
#include <stdlib.h>
#include <time.h>
#include "boids.h"

// Globally declare the boid array in WebAssembly linear memory space
static Boid boid_arr[MAX_BOIDS];
static int active_boid_count = 0;

// Tell the compiler to keep this function alive for JavaScript exposure
EMSCRIPTEN_KEEPALIVE
void init_simulation(int boid_count, int width, int height) {
    srand(time(NULL));
    if (boid_count > MAX_BOIDS) boid_count = MAX_BOIDS;
    active_boid_count = boid_count;

    for (int i = 0; i < boid_count; i++) {
        boid_random_init(&boid_arr[i], width, height);
    }
}

EMSCRIPTEN_KEEPALIVE
void step_simulation(float separation_strength, float cohesion_strength, float alignment_strength, int width, int height, float dt) {
    static Group groups[MAX_BOIDS];
    float group_radius = 50.0f; 
    float max_speed = 150.0f;

    int group_count = find_groups(boid_arr, active_boid_count, group_radius, groups);

    for (int g = 0; g < group_count; g++) {
        Group current_group = groups[g];
        Vector2 avg_pos = find_average_pos(boid_arr, &current_group);
        Vector2 avg_vel = find_average_vel(boid_arr, &current_group);

        for (int m = 0; m < current_group.count; m++) {
            int boid_id = current_group.members[m];

            Vector2 separation_force = separation(boid_arr, &current_group, m, separation_strength);
            Vector2 cohesion_force = cohesion(boid_arr, &current_group, m, avg_pos, cohesion_strength);
            Vector2 alignment_force = alignment(boid_arr, &current_group, m, avg_vel, alignment_strength);

            Vector2 total_force = vector_add(vector_add(separation_force, cohesion_force), alignment_force);
            
            boid_update(&boid_arr[boid_id], total_force, dt);

            // Cap Speed
            float current_speed = vector_length(boid_arr[boid_id].velocity);
            if (current_speed > max_speed) {
                boid_arr[boid_id].velocity = vector_multiply(vector_normalize(boid_arr[boid_id].velocity), max_speed);
            }
            check_boundaries(&boid_arr[boid_id], width, height);
        }
    }
}

EMSCRIPTEN_KEEPALIVE
Boid* get_boids_buffer() {
    return boid_arr;
}