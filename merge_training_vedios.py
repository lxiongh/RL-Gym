import os
import json
import sys

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print(sys.argv)
        print("Usage: python vedio_dir output_name")
        sys.exit()
    
    vedio_dir = sys.argv[1]
    output_name = sys.argv[2]
    list_dirs = os.walk(vedio_dir)

    tmp_dir = '/tmp/openai'
    os.system('rm -rf %s' % tmp_dir)
    os.mkdir(tmp_dir)

    names = []
    for root, dirs, files in list_dirs:
        files = [f for f in files if f.endswith('.mp4')]
        for f in files:
            name = f.replace('.mp4', '')
            names.append(name)
            meta_path = '%s/%s.meta.json' % (vedio_dir, name)
            fid = open(meta_path, 'r')
            meta = json.loads(fid.readline())
            fid.close()
            i_eps = meta['episode_id']

            fsrt = open('%s/%s.srt' % (tmp_dir, name), 'w')
            fsrt.write('1\n00:00:00,0 --> 00:10:00,0\nEpisode %d\n' % i_eps)
            fsrt.close()

            # merge mp4 and srt
            cmd = 'ffmpeg -i %s/%s.mp4 -vf subtitles=%s/%s.srt %s/%s.mp4' % \
                (vedio_dir, name, tmp_dir, name, tmp_dir, name)
            os.system(cmd)

    flist = open('%s/flist.txt' % tmp_dir, 'w+')
    names.sort()
    for name in names:
        flist.write("file '%s/%s.mp4'\n" % (tmp_dir, name))

    flist.close()

    # merger vedio
    cmd = 'ffmpeg -f concat -safe 0 -i %s/flist.txt -c copy %s' % (tmp_dir, output_name)
    print(cmd)
    os.system(cmd)

    # # convert to gif
    # # 1. 生成调色板
    # cmd = 'ffmpeg -y -i output.mp4 -vf palettegen palette.png'
    # os.system(cmd)
    # # 2. 根据调色板生成 GIF, 这样起到了压缩的作用
    # cmd = 'ffmpeg -y -i output.mp4 -i palette.png -filter_complex paletteuse -r 10 output.gif'
    # print(cmd)
    # os.system(cmd)