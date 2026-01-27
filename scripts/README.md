# Lalo Scripts

This directory contains utility scripts for working with audiobooks created by Lalo.

## Available Scripts

### merge_m4b.sh

Merge multiple M4B audiobook files into a single file while preserving all chapter markers.

**Purpose**: When you convert a large book in parts (e.g., chapters 1-10, 11-20, 21-30), this script combines them into one complete audiobook with all chapters properly marked.

**Usage**:
```bash
./scripts/merge_m4b.sh [OPTIONS] OUTPUT_FILE INPUT_FILE1 INPUT_FILE2 [INPUT_FILE3 ...]
```

**Options**:
- `-v, --verbose` - Show detailed ffmpeg output
- `-n, --dry-run` - Preview the merge without actually doing it
- `-f, --force` - Overwrite output file if it exists
- `-k, --keep-temp` - Keep temporary files for debugging
- `-h, --help` - Show help message

**Examples**:

Merge three parts:
```bash
./scripts/merge_m4b.sh complete_book.m4b part1.m4b part2.m4b part3.m4b
```

Merge using wildcards:
```bash
./scripts/merge_m4b.sh complete_book.m4b book_part*.m4b
```

Preview before merging:
```bash
./scripts/merge_m4b.sh --dry-run output.m4b part1.m4b part2.m4b
```

**Features**:
- ✓ Preserves all chapter markers with correct timestamps
- ✓ Uses first file's metadata (title, artist) as source of truth
- ✓ Warns about metadata mismatches between files
- ✓ Fast concatenation without re-encoding
- ✓ Validates all input files before processing

**Requirements**:
- `ffmpeg` - Audio/video processing (already required by Lalo)
- `ffprobe` - Media file analysis (comes with ffmpeg)
- `jq` - JSON parsing
- `bc` - Arbitrary precision calculator (used for time and duration calculations)
- `realpath` - Resolve absolute, canonical file paths (usually part of coreutils)

**Output**:
The script creates a single M4B file with:
- Combined audio from all input files (in order)
- All chapters from all files with adjusted timestamps
- Metadata from the first file
- Optimized for streaming (movflags +faststart)

**Metadata Handling**:
The script uses the first input file as the "source of truth" for metadata. If other files have different titles or authors, you'll see warnings like:

```
WARNING: File part2.m4b has different title: "My Book - Part 2"
  Using source file's title: "My Book"
```

This is expected when merging parts of the same book. The final output will have consistent metadata from the first file.
