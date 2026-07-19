#include "boids.h"
#include <math.h>

void boid_random_init(Boid *boid, int width, int height)
{
    boid->position = (Vector2){rand() % width,rand() % height};
    boid->velocity = (Vector2){rand() % 100 -100,rand() % 100 - 100};
    boid->is_grouped = false;
}

void boid_update(Boid *boid, Vector2 change, float dt)
{
    boid->velocity = vector_add(boid->velocity, vector_multiply(change, dt));

    boid->position = vector_add(boid->position,vector_multiply(boid->velocity, dt));
}


void check_boundaries(Boid *boid, int width, int height)
{
    if (boid->position.x < 0)
        boid->position.x = width;

    if (boid->position.x > width)
        boid->position.x = 0;

    if (boid->position.y < 0)
        boid->position.y = height;

    if (boid->position.y > height)
        boid->position.y = 0;
}

Vector2 separation(Boid *boids, Group *group, int member_index, float separation_strength) {
    if (group->count <= 1) return (Vector2){0.0f, 0.0f};

    Vector2 vec = {0.0f, 0.0f};
    int current_boid = group->members[member_index];

    for (int i = 0; i < group->count; i++) {
        if (i == member_index) continue;
        int other_boid = group->members[i];

        Vector2 displacement = vector_subtract(boids[current_boid].position, boids[other_boid].position);
        float mod_r = vector_length(displacement);

        if (mod_r > 0.0f && mod_r < 40.0f) {
            vec = vector_add(vec, vector_normalize(displacement));
        }
    }
    return vector_multiply(vec, separation_strength);
}


Vector2 cohesion(Boid *boids, Group *group, int member_index, Vector2 avg_pos, float cohesion_strength) {
    if (group->count <= 1) return (Vector2){0.0f, 0.0f};

    int current_boid = group->members[member_index];
    Vector2 difference = vector_subtract(avg_pos, boids[current_boid].position);

    return vector_multiply(difference, cohesion_strength);
}

Vector2 alignment(Boid *boids, Group *group, int member_index, Vector2 avg_vel, float alignment_strength) {
    if (group->count <= 1) return (Vector2){0.0f, 0.0f};

    int current_boid = group->members[member_index];
    Vector2 difference = vector_subtract(avg_vel, boids[current_boid].velocity);

    return vector_multiply(difference, alignment_strength);
}

Vector2 find_average_pos(Boid *boids, Group *group){
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (int i = 0; i < group->count; i++){
        sum_x += boids[group->members[i]].position.x;
        sum_y += boids[group->members[i]].position.y;
    }
        
    float avg_x = sum_x / group->count;
    float avg_y = sum_y / group->count;

    Vector2 avg_pos = (Vector2){avg_x, avg_y};
    return avg_pos;
}

Vector2 find_average_vel(Boid *boids, Group *group){
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (int i = 0; i<group->count; i++){
        sum_x += boids[group->members[i]].velocity.x;
        sum_y += boids[group->members[i]].velocity.y;
    }
        
    float avg_x = sum_x / group->count;
    float avg_y = sum_y / group->count;

    Vector2 avg_vel = (Vector2){avg_x, avg_y};
    return avg_vel;
}

int find_groups(Boid *boids, int boid_count, float group_radius, Group *groups_out) {
    // FIX: Use a static matrix with MAX_BOIDS bounds so it lives in safe linear memory instead of overflowing the WASM stack!
    static bool adj[MAX_BOIDS][MAX_BOIDS]; 

    // Explicitly reset the matrix paths for the active frame count
    for (int i = 0; i < boid_count; i++) {
        for (int j = 0; j < boid_count; j++) {
            adj[i][j] = false;
        }
    }

    // Build the adjacency matrix connections
    for (int i = 0; i < boid_count; i++) {
        for (int j = i + 1; j < boid_count; j++) {
            Vector2 displacement = vector_subtract(boids[i].position, boids[j].position);
            float mod_r = vector_length(displacement);
            if (mod_r < group_radius) {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
    }

    // Allocate tracking memory buffers statically
    static bool visited[MAX_BOIDS];
    static int queue[MAX_BOIDS];
    
    // Clear visited tracker
    for (int i = 0; i < boid_count; i++) {
        visited[i] = false;
        boids[i].is_grouped = false; // Reset grouped flag per loop frame safely
    }

    int head = 0, tail = 0;
    int group_count = 0;

    for (int i = 0; i < boid_count; i++) {
        if (visited[i]) continue;

        Group *new_group = &groups_out[group_count++];
        new_group->count = 0;

        visited[i] = true;
        queue[tail++] = i;

        while (head < tail) {
            int current = queue[head++];
            new_group->members[new_group->count++] = current;

            for (int j = 0; j < boid_count; j++) {
                if (adj[current][j] && !visited[j]) {
                    visited[j] = true;
                    queue[tail++] = j;
                }
            }
        }

        if (new_group->count > 1) {
            for (int m = 0; m < new_group->count; m++) {
                boids[new_group->members[m]].is_grouped = true;
            }
        }
    }

    return group_count;
}