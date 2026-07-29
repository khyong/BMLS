#!/usr/bin/env python3
"""Resize iNaturalist2018 images to an aspect-preserving JPEG dataset.

The existing ImageNet-style list files contain paths such as
``train_val2018/Aves/...jpg`` while the source files live below ``images/``.
This script preserves that layout under the output root, so the existing
training and validation list files continue to work unchanged. The shorter
image side is resized to ``--size`` while the original aspect ratio is kept.
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

from PIL import Image, ImageOps


_WORKER_SOURCE_ROOT = None
_WORKER_OUTPUT_ROOT = None
_WORKER_SIZE = None
_WORKER_QUALITY = None
_WORKER_OVERWRITE = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a resized iNaturalist2018 image directory.'
    )
    parser.add_argument(
        '--source-root',
        type=Path,
        default=Path('/data/hoyong/iNaturalist2018'),
        help='Original dataset root containing images/ (default: %(default)s)',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help=(
            'Destination dataset root. By default, ina2018_short256 is created '
            'beside --source-root.'
        ),
    )
    parser.add_argument(
        '--list-root',
        type=Path,
        default=Path('/data/hoyong/data_txt'),
        help='Directory containing iNaturalist18_train/val.txt',
    )
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='Output length of the shorter image side (default: %(default)s)',
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=90,
        help='JPEG quality from 1 to 100 (default: %(default)s)',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=min(16, os.cpu_count() or 1),
        help='Resize worker processes (default: %(default)s)',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Recreate images that already exist and are non-empty.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process only the first N unique images (useful for testing).',
    )
    return parser.parse_args()


def read_relative_paths(list_files):
    paths = []
    seen = set()
    for list_file in list_files:
        if not list_file.is_file():
            raise FileNotFoundError('List file does not exist: {}'.format(list_file))
        with list_file.open('r', encoding='utf-8') as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    relative_text, _ = line.rsplit(maxsplit=1)
                except ValueError as exc:
                    raise ValueError(
                        'Invalid entry in {} at line {}: {!r}'.format(
                            list_file, line_number, raw_line.rstrip('\n')
                        )
                    ) from exc
                relative_path = Path(relative_text)
                if relative_path.is_absolute() or '..' in relative_path.parts:
                    raise ValueError(
                        'Unsafe relative path in {} at line {}: {}'.format(
                            list_file, line_number, relative_path
                        )
                    )
                normalized = relative_path.as_posix()
                if normalized not in seen:
                    seen.add(normalized)
                    paths.append(normalized)
    return paths


def init_worker(source_root, output_root, size, quality, overwrite):
    global _WORKER_SOURCE_ROOT
    global _WORKER_OUTPUT_ROOT
    global _WORKER_SIZE
    global _WORKER_QUALITY
    global _WORKER_OVERWRITE

    _WORKER_SOURCE_ROOT = Path(source_root)
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_SIZE = int(size)
    _WORKER_QUALITY = int(quality)
    _WORKER_OVERWRITE = bool(overwrite)


def resize_one(relative_text):
    relative_path = Path(relative_text)
    source_path = _WORKER_SOURCE_ROOT  / relative_path
    output_path = _WORKER_OUTPUT_ROOT  / relative_path

    if (
        not _WORKER_OVERWRITE
        and output_path.is_file()
        and output_path.stat().st_size > 0
    ):
        return 'skipped', relative_text, None

    temporary_path = output_path.with_name(
        '{}.tmp-{}'.format(output_path.name, os.getpid())
    )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open('rb') as source_handle:
            with Image.open(source_handle) as source_image:
                image = ImageOps.exif_transpose(source_image).convert('RGB')
                width, height = image.size
                scale = _WORKER_SIZE / float(min(width, height))
                resized_size = (
                    max(_WORKER_SIZE, int(round(width * scale))),
                    max(_WORKER_SIZE, int(round(height * scale))),
                )
                image = image.resize(
                    resized_size,
                    resample=Image.Resampling.LANCZOS,
                )
                image.save(
                    temporary_path,
                    format='JPEG',
                    quality=_WORKER_QUALITY,
                    optimize=False,
                )
        os.replace(temporary_path, output_path)
        return 'resized', relative_text, None
    except Exception as exc:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        return 'failed', relative_text, '{}: {}'.format(type(exc).__name__, exc)


def iter_with_progress(results, total):
    try:
        from tqdm import tqdm

        yield from tqdm(results, total=total, unit='image')
    except ImportError:
        for index, result in enumerate(results, start=1):
            if index == 1 or index % 1000 == 0 or index == total:
                print('Processed {}/{} images'.format(index, total), flush=True)
            yield result


def main():
    args = parse_args()
    if args.size < 1:
        raise ValueError('--size must be positive')
    if not 1 <= args.quality <= 100:
        raise ValueError('--quality must be between 1 and 100')
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')

    source_root = args.source_root.expanduser().resolve()
    if args.output_root is None:
        output_root = source_root.parent / 'ina2018_short256'
    else:
        output_root = args.output_root.expanduser().resolve()
    list_root = args.list_root.expanduser().resolve()

    if not (source_root).is_dir():
        raise FileNotFoundError(
            'Source images directory does not exist: {}'.format(
                source_root / 'images'
            )
        )
    if output_root == source_root:
        raise ValueError('--output-root must be different from --source-root')

    list_files = [
        list_root / 'iNaturalist18_train.txt',
        list_root / 'iNaturalist18_val.txt',
    ]
    relative_paths = read_relative_paths(list_files)
    if args.limit is not None:
        relative_paths = relative_paths[:args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    print('Source root : {}'.format(source_root))
    print('Output root : {}'.format(output_root))
    print('Output size : shorter side {}, aspect ratio preserved'.format(args.size))
    print('JPEG quality: {}'.format(args.quality))
    print('Workers     : {}'.format(args.workers))
    print('Images      : {}'.format(len(relative_paths)))

    counts = {'resized': 0, 'skipped': 0, 'failed': 0}
    failures = []
    with mp.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(
            str(source_root),
            str(output_root),
            args.size,
            args.quality,
            args.overwrite,
        ),
    ) as pool:
        results = pool.imap_unordered(
            resize_one, relative_paths, chunksize=16
        )
        for status, relative_text, error in iter_with_progress(
            results, len(relative_paths)
        ):
            counts[status] += 1
            if error is not None:
                failures.append('{}\t{}'.format(relative_text, error))

    failure_log = output_root / 'resize_failures.txt'
    if failures:
        failure_log.write_text('\n'.join(failures) + '\n', encoding='utf-8')
    elif failure_log.exists():
        failure_log.unlink()

    print(
        'Finished: resized={resized}, skipped={skipped}, failed={failed}'.format(
            **counts
        )
    )
    if failures:
        print('Failure details: {}'.format(failure_log))
        raise SystemExit(1)
    print('Set BMLSBN_INAT_ROOT={}'.format(output_root))


if __name__ == '__main__':
    main()
