base_url = "https://williamtheisen.com/nd-cse-10124-lectures/Lecture_Images/Lecture01"

start = 1
end = 19  # non-inclusive, i.e. 19–30

for i in range(start, end):
    print(f'''<div class="thumbnail">
    <img src="{base_url}/slide-{i:03d}.png" class="img-responsive"/>
</div>''')