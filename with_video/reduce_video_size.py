import os

os.makedirs("./output_videos_processed", exist_ok=True)
commands = []
for i in os.listdir("./output_videos/"):
    video_path = os.path.join("./output_videos", i)
    out_path = os.path.join("./output_videos_processed", i)
    cmd = "ffmpeg -i {} -c:v libx264 -s 1920x1080 -b:v 1.5M -c:a aac -b:a 128k -crf 23 {}".format(
        video_path, out_path
    )
    commands.append((cmd,))
from multiprocessing import Pool

pool = Pool(12)
pool.starmap(os.system, commands)
pool.join()
pool.close()
