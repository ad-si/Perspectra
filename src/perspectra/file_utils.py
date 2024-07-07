"""
Renames the images in a directory according to their pages number.
The assumend layout is `cover -> odd pages -> even pages reversed`.
The cover gets the page number 0.

For example a book with 9 content pages and a cover starting at image 23:

- img_23.jpg => 0

- img_24.jpg => 1
- img_25.jpg => 3
- img_26.jpg => 5
- img_27.jpg => 7
- img_28.jpg => 9

- img_29.jpg => 8
- img_30.jpg => 6
- img_31.jpg => 4
- img_32.jpg => 2
"""

from pathlib import Path


def getTempPath (file_path):
    return file_path.with_name(
        f'temporary-name-to-avoid-collisions_{file_path.name}'
    )


def renumber_pages (book_directory = '.'):
    # Configuration
    shall_run_dry = True
    includes_cover = True

    valid_file_endings = ('jpeg', 'jpg', 'png', 'tiff', 'tif', 'gif')
    book_dir_path = Path(book_directory).resolve()
    images = [
        entry for entry in book_dir_path.iterdir()
        if entry.suffix.lower()[1:] in valid_file_endings
    ]

    if includes_cover:
        num_pages = len(images)
        num_content_pages = num_pages - 1
        split_point = int(num_content_pages / 2) + 1
        last_page_is_odd = num_pages % 2 == 0

        if last_page_is_odd:
            split_point += 1

        odd_pages = images[1:split_point]
        even_pages = images[split_point:][::-1]

        sorted_images = [images[0]] + [img
            for tup in zip(odd_pages, even_pages)
            for img in tup
        ]
    else:
        raise TypeError('TODO: Implement renaming if pages don\'t include a cover')

    print(f'In "{book_dir_path}" move:\n')

    for (index, file_path) in enumerate(sorted_images):
        temp_path = getTempPath(file_path)

        print(f'\t{file_path.name} -> {temp_path.name}', end='')
        if not shall_run_dry:
            file_path.rename(temp_path)
        print(' ✔︎')

    print()

    for (index, file_path) in enumerate(sorted_images):
        temp_path = getTempPath(file_path)
        name_length = len(str(num_pages))
        output_path = temp_path.with_name(f'{index:0{name_length}d}.jpg')

        print(f'\t{temp_path.name} -> {output_path.name}', end='')
        if not shall_run_dry:
            temp_path.rename(output_path)
        print(' ✔︎')
