#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <stdio.h>
#include <stdlib.h>
#include "boids.h"



int main(int argc, char* argv[]) {

    SDL_Window *window;                    // Declare a pointer

    int width = 1920;
    int height = 1080;

    SDL_Init(SDL_INIT_VIDEO);              // Initialize SDL3

    // Create an application window with the following settings:
    window = SDL_CreateWindow(
        "An SDL3 window",                  // window title
        width,                               // width, in pixels
        height,                               // height, in pixels
        SDL_WINDOW_OPENGL                  // flags - see below
    );

    // Check that the window was successfully created
    if (window == NULL) {
        // In the case that the window could not be made...
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not create window: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, NULL);

    if (renderer == NULL)
    {
        SDL_Log("Renderer creation failed: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }



    bool done = false;
    float speed = 5.0f;

    int boid_count = 150;

    

    Boid boid_arr[boid_count];

    for(int i = 0; i<boid_count; i++){
        boid_random_init(&boid_arr[i], width, height);
    }

    Uint64 last = SDL_GetPerformanceCounter();
    while (!done) {
        SDL_Event event;

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                done = true;
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        // Do game logic, present a frame, etc.

        Uint64 now = SDL_GetPerformanceCounter();
        float dt = 1 * (float)(now - last) / SDL_GetPerformanceFrequency();
        last = now;
        float group_radius = 80.0f;
        float max_speed = 200.0f;
        float cohesion_strength = 0.5f;
        float separation_strength = 150.0f;
        float alignment_strength = 1.5f;

        Group groups[MAX_BOIDS];
        int total_groups = find_groups(boid_arr, boid_count, group_radius, groups);

        
        for (int g = 0; g < total_groups; g++) {
            Group current_group = groups[g];

            Vector2 avg_pos = find_average_pos(boid_arr, &current_group);
            Vector2 avg_vel = find_average_vel(boid_arr, &current_group);

            for (int m = 0; m < current_group.count; m++) {
                int boid_id = current_group.members[m];

                check_boundaries(&boid_arr[boid_id], width, height);

                Vector2 separation_force = separation(boid_arr, &current_group, m, separation_strength);            
                Vector2 cohesion_force = cohesion(boid_arr, &current_group, m, avg_pos, cohesion_strength);
                Vector2 alignment_force = alignment(boid_arr, &current_group, m, avg_vel, alignment_strength);

                Vector2 total_force = vector_add(vector_add(separation_force, cohesion_force), alignment_force);
                
                boid_update(&boid_arr[boid_id], total_force, dt);

                // Speed Capping
                float current_speed = vector_length(boid_arr[boid_id].velocity);
                if (current_speed > max_speed) {
                    boid_arr[boid_id].velocity = vector_multiply(vector_normalize(boid_arr[boid_id].velocity), max_speed);
                }
            }
        }

        float boid_size = 6.0f; 
        for (int i = 0; i < boid_count; i++) {
            SDL_FRect boid_rect = {
                .x = boid_arr[i].position.x - (boid_size / 2.0f), 
                .y = boid_arr[i].position.y - (boid_size / 2.0f),
                .w = boid_size,
                .h = boid_size
            };
            if (boid_arr[i].is_grouped) {
                SDL_SetRenderDrawColor(renderer, 255, 50, 50, 255);
            } else {
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            }
            SDL_RenderFillRects(renderer, &boid_rect, 1);
        }


        SDL_RenderPresent(renderer);
    }

    // Close and destroy the window
    SDL_DestroyWindow(window);

    // Clean up
    SDL_Quit();
    return 0;


}
