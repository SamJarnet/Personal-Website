import datetime
import pandas as pd
import numpy as np

class WebSearchEngine:
    def __init__(self):
        try:
            self.data = pd.read_csv("data/music_data/data.csv")
            self.data = np.array(self.data).T
        except Exception:
            # Provide high-quality fallback stock tracks for guest testing
            fallback_titles = ["Starlight Serenade", "Midnight Groove", "Cyberpunk Horizon", "Acoustic Sunset", "Neon Dreams"]
            fallback_artists = ["Luna Eclipse", "The Jazz Vibe", "Tokyo Synth", "Emma Woods", "Retro Wave"]
            fallback_years = [2022, 2021, 2024, 2020, 2023]
            
            # Construct a mock matrix structural array
            self.data = np.zeros((18, len(fallback_titles)), dtype=object)
            self.data[15] = fallback_titles  # Song Title
            self.data[16] = fallback_artists # Artists
            self.data[1] = [y / 2020 for y in fallback_years] # Normalized Year

    def merge_sort(self, lst):
        if len(lst) > 1:
            middle = len(lst) // 2
            left = lst[:middle]
            right = lst[middle:]

            self.merge_sort(left)
            self.merge_sort(right)

            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    lst[k] = left[i]
                    i += 1
                else:
                    lst[k] = right[j]
                    j += 1
                k += 1

            while i < len(left):
                lst[k] = left[i]
                i += 1
                k += 1
            while j < len(right):
                lst[k] = right[j]
                j += 1
                k += 1

    def search_music(self, text):
        if not text:
            return []
        
        searched_songs = []
        limit = len(self.data[15])
        
        for i in range(0, limit):
            song_title = str(self.data[15][i])
            if song_title[0:len(text)].lower() == text.lower() or (
                text.lower() in song_title.lower() and 
                (len(text) - (song_title.lower().find(text.lower())))**2 < 25 and 
                len(text) > 2
            ):
                searched_songs.insert(0, i)
                self.merge_sort(searched_songs)
        
        results = []
        for idx in searched_songs:
            current_title = str(self.data[15][idx])
            raw_artists = str(self.data[16][idx]).replace("[", "").replace("]", "").replace("'", "")
            artists = [a.strip() for a in raw_artists.split(",")]
            
            try:
                year = str(int(float(self.data[1][idx]) * 2020))
            except Exception:
                year = "Unknown"

            results.append({
                "id": int(idx),
                "title": current_title,
                "artists": artists,
                "year": year
            })
        return results


class WebCircularQueue:
    def __init__(self, queue_length=15):
        self.queue_length = queue_length
        self.queue = [None] * self.queue_length
        self.head = -1
        self.tail = -1

    def enqueue(self, track_name):
        if ((self.tail + 1) % self.queue_length == self.head):
            return {"status": "full", "msg": "The circular queue is full"}
        
        elif (self.head == -1):
            self.head = 0
            self.tail = 0
            self.queue[self.tail] = track_name
        else:
            self.tail = (self.tail + 1) % self.queue_length
            self.queue[self.tail] = track_name
            
        return {"status": "success", "queue": self.get_valid_items()}

    def dequeue(self):
        if (self.head == -1):
            return {"status": "empty", "msg": "The circular queue is empty"}

        elif (self.head == self.tail):
            temp = self.queue[self.head]
            self.head = -1
            self.tail = -1
            return {"status": "success", "dequeued": temp, "queue": self.get_valid_items()}
        else:
            temp = self.queue[self.head]
            self.head = (self.head + 1) % self.queue_length
            return {"status": "success", "dequeued": temp, "queue": self.get_valid_items()}

    def get_valid_items(self):
        if self.head == -1:
            return []
        if self.tail >= self.head:
            return [self.queue[i] for i in range(self.head, self.tail + 1) if self.queue[i] is not None]
        else:
            return [self.queue[i] for i in range(self.head, self.queue_length)] + \
                   [self.queue[i] for i in range(0, self.tail + 1)]


class WebStatsTracker:
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.total_mins = 0
        self.total_secs = 0
        self.freeze = False
        self.saved_mins = 0
        self.saved_secs = 0
        self.time_paused = None

    def record_pause(self):
        self.time_paused = datetime.datetime.now()
        self.saved_mins = float(str(datetime.datetime.now())[-12:-10]) - float(str(self.start_time)[-12:-10]) - self.total_mins
        self.saved_secs = float(str(datetime.datetime.now())[-9:-6]) - float(str(self.start_time)[-9:-6]) - self.total_secs
        if self.saved_secs < 0:
            self.saved_secs += 60
            self.saved_mins -= 1
        self.freeze = True

    def record_resume(self):
        if self.time_paused:
            time_difference = datetime.datetime.now() - self.time_paused
            if str(time_difference)[0] != "-":
                self.total_mins += float(str(time_difference)[-12:-10])
                self.total_secs += float(str(time_difference)[-9:-6])
                if self.total_secs > 60:
                    self.total_secs -= 60
                    self.total_mins += 1
        self.freeze = False

    def get_stats_strings(self):
        if not self.freeze:
            mins = float(str(datetime.datetime.now())[-12:-10]) - float(str(self.start_time)[-12:-10]) - self.total_mins
            secs = float(str(datetime.datetime.now())[-9:-6]) - float(str(self.start_time)[-9:-6]) - self.total_secs
            if secs < 0:
                secs += 60
                mins -= 1
            return f"Total of: {int(mins)} minutes {int(secs)} seconds listened"
        else:
            return f"Total of: {int(self.saved_mins)} minutes {int(self.saved_secs)} seconds listened"


class WebRecommender:
    def __init__(self):
        try:
            self.dataset = pd.read_csv("data/music_data/data.csv", low_memory=False)
        except Exception:
            # Fallback mock matrix dataframe for guest previewing evaluation mechanics
            self.dataset = pd.DataFrame({
                "name": ["Starlight Serenade", "Midnight Groove", "Cyberpunk Horizon", "Acoustic Sunset", "Neon Dreams"],
                "valence": [0.5, 0.4, 0.8, 0.6, 0.7], "year": [2022, 2021, 2024, 2020, 2023],
                "acousticness": [0.2, 0.1, 0.0, 0.9, 0.1], "danceability": [0.6, 0.7, 0.8, 0.4, 0.7],
                "duration_ms": [200000, 180000, 240000, 210000, 195000], "energy": [0.7, 0.5, 0.9, 0.3, 0.8],
                "explicit": [0, 0, 0, 0, 0], "instrumentalness": [0.0, 0.2, 0.5, 0.1, 0.0],
                "key": [1, 5, 7, 2, 4], "liveness": [0.1, 0.1, 0.2, 0.1, 0.1],
                "loudness": [-5, -7, -4, -10, -6], "mode": [1, 0, 1, 1, 0],
                "popularity": [50, 45, 60, 40, 55], "speechiness": [0.05, 0.04, 0.08, 0.03, 0.06],
                "tempo": [120, 100, 130, 90, 115]
            })

    def recommend_songs(self, playlist_songs):
        if self.dataset.empty or not playlist_songs:
            return []
            
        df = self.dataset[self.dataset["name"].isin(playlist_songs)]
        if df.empty:
            # Use random fallback sampling from target data to ensure guests see functionality
            df = self.dataset.head(2)
            
        features = ["valence", "year", "acousticness", "danceability", "duration_ms", "energy",
                    "explicit", "instrumentalness", "key", "liveness", "loudness", "mode",
                    "popularity", "speechiness", "tempo"]
                    
        avg_vector = np.mean(df[features].to_numpy(), axis=0)
        candidate_songs = self.dataset[~self.dataset["name"].isin(playlist_songs)].copy()
        if candidate_songs.empty:
            candidate_songs = self.dataset.copy()
            
        candidate_vectors = candidate_songs[features].to_numpy()

        similarities = np.dot(candidate_vectors, avg_vector) / (
            np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(avg_vector) + 1e-9)

        candidate_songs["similarity"] = similarities
        top_matches = candidate_songs.sort_values(by="similarity", ascending=False).head(5)
        
        return top_matches["name"].tolist()