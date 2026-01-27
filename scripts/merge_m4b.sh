#!/usr/bin/env bash
#
# merge_m4b.sh - Merge multiple M4B audiobook files while preserving chapters
#
# Usage: merge_m4b.sh [OPTIONS] OUTPUT_FILE INPUT_FILE1 INPUT_FILE2 [INPUT_FILE3 ...]
#
# This script merges multiple M4B audiobook files into a single file while:
# - Preserving all chapter markers with correct timestamps
# - Using the first file's metadata as source of truth
# - Warning about metadata mismatches between files
# - Concatenating audio without re-encoding (fast!)
#

set -euo pipefail

# Colors for output using tput
if [[ -t 1 ]] && command -v tput &> /dev/null && tput setaf 1 &> /dev/null; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    CYAN=$(tput setaf 6)
    BOLD=$(tput bold)
    NC=$(tput sgr0) # Reset
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# Default options
VERBOSE=false
DRY_RUN=false
FORCE=false
KEEP_TEMP=false

# Usage information
usage() {
    local exit_code="${1:-0}"
    cat << EOF
${BOLD}M4B Merge Tool${NC} - Part of the Lalo audiobook converter

${BOLD}USAGE:${NC}
    $(basename "$0") [OPTIONS] OUTPUT_FILE INPUT_FILE1 INPUT_FILE2 [INPUT_FILE3 ...]

${BOLD}DESCRIPTION:${NC}
    Merges multiple M4B audiobook files into a single file while preserving
    all chapter markers with properly adjusted timestamps.

${BOLD}OPTIONS:${NC}
    -v, --verbose       Show detailed ffmpeg output
    -n, --dry-run       Show what would be done without actually merging
    -f, --force         Overwrite output file if it exists
    -k, --keep-temp     Keep temporary files for debugging
    -h, --help          Show this help message

${BOLD}EXAMPLES:${NC}
    # Merge three parts into one complete audiobook
    $(basename "$0") complete.m4b part1.m4b part2.m4b part3.m4b

    # Merge with wildcard expansion
    $(basename "$0") complete.m4b book_part*.m4b

    # Dry run to preview the merge
    $(basename "$0") --dry-run output.m4b file1.m4b file2.m4b

${BOLD}REQUIREMENTS:${NC}
    - ffmpeg and ffprobe must be installed
    - jq must be installed for JSON parsing
    - All input files must be M4B format

${BOLD}METADATA HANDLING:${NC}
    The first input file is used as the "source of truth" for metadata.
    If subsequent files have different metadata, warnings will be displayed.

${BOLD}EXIT CODES:${NC}
    0  Success
    1  Invalid arguments
    2  Missing dependencies
    3  Invalid input files
    4  Processing error
    5  Output error

EOF
    exit "$exit_code"
}

# Error handling
error() {
    printf "${RED}ERROR:${NC} %b\n" "$1" >&2
    exit "${2:-1}"
}

warn() {
    printf "${YELLOW}WARNING:${NC} %b\n" "$1" >&2
}

info() {
    printf "${CYAN}%b${NC}\n" "$1"
}

success() {
    printf "${GREEN}✓${NC} %b\n" "$1"
}

# Escape special characters for FFMETADATA1 format
escape_ffmetadata() {
    local value="$1"
    # Escape special characters: = ; # \ and newlines
    value="${value//\\/\\\\}"  # Escape backslashes first
    value="${value//=/\\=}"    # Escape equals
    value="${value//;/\\;}"    # Escape semicolons
    value="${value//#/\\#}"    # Escape hash
    value="${value//$'\n'/\\n}" # Escape newlines
    value="${value//$'\r'/\\r}" # Escape carriage returns
    echo "$value"
}

# Check dependencies
check_dependencies() {
    local missing=()

    if ! command -v ffmpeg &> /dev/null; then
        missing+=("ffmpeg")
    fi

    if ! command -v ffprobe &> /dev/null; then
        missing+=("ffprobe")
    fi

    if ! command -v jq &> /dev/null; then
        missing+=("jq")
    fi

    if ! command -v bc &> /dev/null; then
        missing+=("bc")
    fi

    if ! command -v realpath &> /dev/null; then
        missing+=("realpath")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing required dependencies: ${missing[*]}\n\nPlease install these tools using your package manager.\n\nExamples:\n  Ubuntu/Debian: sudo apt-get install ffmpeg jq bc coreutils\n  macOS (Homebrew): brew install ffmpeg jq coreutils" 2
    fi
}

# Extract metadata from M4B file
get_metadata() {
    local file="$1"
    local key="$2"

    ffprobe -v quiet -print_format json -show_format "$file" | \
        jq -r ".format.tags.${key} // empty"
}

# Get duration in milliseconds
get_duration_ms() {
    local file="$1"

    local duration_sec
    duration_sec=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$file")

    # Convert to milliseconds and round
    echo "$duration_sec * 1000 / 1" | bc
}

# Extract chapters from M4B file
get_chapters() {
    local file="$1"

    ffprobe -v quiet -print_format json -show_chapters "$file"
}

# Format duration from milliseconds to human readable
format_duration() {
    local ms=$1
    local seconds=$((ms / 1000))
    local minutes=$((seconds / 60))
    local hours=$((minutes / 60))

    minutes=$((minutes % 60))
    seconds=$((seconds % 60))

    if [[ $hours -gt 0 ]]; then
        printf "%dh %02dm %02ds" $hours $minutes $seconds
    else
        printf "%dm %02ds" $minutes $seconds
    fi
}

# Format file size
format_size() {
    local size=$1

    if [[ $size -ge 1073741824 ]]; then
        echo "$(echo "scale=1; $size / 1073741824" | bc) GB"
    elif [[ $size -ge 1048576 ]]; then
        echo "$(echo "scale=1; $size / 1048576" | bc) MB"
    else
        echo "$(echo "scale=1; $size / 1024" | bc) KB"
    fi
}

# Validate M4B file
validate_m4b() {
    local file="$1"

    # Check if file exists
    if [[ ! -f "$file" ]]; then
        return 1
    fi

    # Check format
    local format
    format=$(ffprobe -v quiet -print_format json -show_format "$file" | \
        jq -r '.format.format_name // empty')

    # M4B files typically show as "mov,mp4,m4a,3gp,3g2,mj2"
    if [[ ! "$format" =~ (mov|mp4|m4a) ]]; then
        return 1
    fi

    return 0
}

# Parse command line arguments
parse_args() {
    # First pass: check for --help before validating argument count
    for arg in "$@"; do
        if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
            usage 0
        fi
    done

    if [[ $# -lt 3 ]]; then
        error "Insufficient arguments. Need OUTPUT_FILE and at least 2 INPUT_FILEs.\nUse --help for usage information." 1
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -k|--keep-temp)
                KEEP_TEMP=true
                shift
                ;;
            -h|--help)
                usage 0
                ;;
            -*)
                error "Unknown option: $1\nUse --help for usage information." 1
                ;;
            *)
                break
                ;;
        esac
    done

    # Remaining arguments are output file and input files
    if [[ $# -lt 3 ]]; then
        error "Insufficient arguments. Need OUTPUT_FILE and at least 2 INPUT_FILEs.\nUse --help for usage information." 1
    fi

    OUTPUT_FILE="$1"
    shift
    INPUT_FILES=("$@")
}

# Main merge function
merge_m4b_files() {
    # Validate inputs
    info "Validating inputs..."

    if [[ ${#INPUT_FILES[@]} -lt 2 ]]; then
        error "Need at least 2 input files to merge" 1
    fi
    success "Found ${#INPUT_FILES[@]} input files"

    # Check if all files exist and are valid M4B
    for file in "${INPUT_FILES[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "File not found: $file" 3
        fi

        if ! validate_m4b "$file"; then
            error "Invalid M4B file: $file" 3
        fi
    done
    success "All files exist and are valid M4B format"

    # Check if output file exists
    if [[ -f "$OUTPUT_FILE" ]] && [[ "$FORCE" != true ]]; then
        error "Output file already exists: $OUTPUT_FILE\nUse --force to overwrite" 5
    fi

    # Check output directory is writable
    local output_dir
    output_dir=$(dirname "$OUTPUT_FILE")
    if [[ ! -w "$output_dir" ]]; then
        error "Cannot write to output directory: $output_dir" 5
    fi
    success "Output path is valid"

    echo

    # Extract metadata from source file (first file)
    info "Extracting metadata..."
    local source_file="${INPUT_FILES[0]}"
    local source_title
    local source_artist
    source_title=$(get_metadata "$source_file" "title")
    source_artist=$(get_metadata "$source_file" "artist")

    echo "Source of truth: $(basename "$source_file")"
    echo "  Title: ${source_title:-<not set>}"
    echo "  Artist: ${source_artist:-<not set>}"

    # Compare metadata across files
    for ((i=1; i<${#INPUT_FILES[@]}; i++)); do
        local file="${INPUT_FILES[$i]}"
        local title
        local artist
        title=$(get_metadata "$file" "title")
        artist=$(get_metadata "$file" "artist")

        if [[ -n "$source_title" && -n "$title" && "$title" != "$source_title" ]]; then
            warn "File $(basename "$file") has different title: \"$title\""
            echo "  Using source file's title: \"$source_title\""
        fi

        if [[ -n "$source_artist" && -n "$artist" && "$artist" != "$source_artist" ]]; then
            warn "File $(basename "$file") has different artist: \"$artist\""
            echo "  Using source file's artist: \"$source_artist\""
        fi
    done

    echo

    # Process each file and collect chapter information
    info "Processing files..."

    local -a durations_ms
    local -a chapter_counts
    local total_chapters=0
    local cumulative_offset=0

    for ((i=0; i<${#INPUT_FILES[@]}; i++)); do
        local file="${INPUT_FILES[$i]}"
        local duration_ms
        local chapters_json
        local chapter_count

        duration_ms=$(get_duration_ms "$file")
        durations_ms+=("$duration_ms")

        chapters_json=$(get_chapters "$file")
        chapter_count=$(echo "$chapters_json" | jq '.chapters | length')
        chapter_counts+=("$chapter_count")
        total_chapters=$((total_chapters + chapter_count))

        local file_num=$((i + 1))
        local offset_str=""
        if [[ $i -gt 0 ]]; then
            offset_str=", offset: $(format_duration "$cumulative_offset")"
        fi

        echo "  [$file_num/${#INPUT_FILES[@]}] $(basename "$file") ($chapter_count chapters, $(format_duration "$duration_ms")$offset_str)"

        cumulative_offset=$((cumulative_offset + duration_ms))
    done

    echo

    if [[ "$DRY_RUN" == true ]]; then
        info "DRY RUN - Would create:"
        echo "Output: $OUTPUT_FILE"
        echo "  Total duration: $(format_duration "$cumulative_offset")"
        echo "  Total chapters: $total_chapters"
        echo "  Files to merge: ${#INPUT_FILES[@]}"
        exit 0
    fi

    # Create temporary directory
    local temp_dir
    temp_dir=$(mktemp -d -t merge_m4b.XXXXXX)

    # Cleanup function
    cleanup() {
        local dir="${1:-}"
        if [[ -n "$dir" && -d "$dir" && "$KEEP_TEMP" != true ]]; then
            rm -rf "$dir"
        elif [[ -n "$dir" && -d "$dir" ]]; then
            info "Temporary files kept in: $dir"
        fi
    }
    trap "cleanup '$temp_dir'" EXIT

    # Create file list for concatenation
    info "Merging audio..."
    local filelist="$temp_dir/filelist.txt"
    for file in "${INPUT_FILES[@]}"; do
        # Use absolute paths and properly escape for concat demuxer
        local abs_path
        abs_path=$(realpath "$file")

        # Validate path doesn't contain newlines or carriage returns (security check)
        if [[ "$abs_path" =~ $'\n' || "$abs_path" =~ $'\r' ]]; then
            error "File path contains newline or carriage return characters (security risk): $file" 3
        fi

        # Escape backslashes and single quotes in the path for concat demuxer
        local escaped_path="$abs_path"
        # First escape backslashes (\ -> \\)
        escaped_path=${escaped_path//\\/\\\\}
        # Then escape single quotes (' -> '\'' as required by ffmpeg concat format)
        escaped_path=${escaped_path//\'/\'\\\'\'}
        printf "file '%s'\n" "$escaped_path" >> "$filelist"
    done

    # Concatenate audio without re-encoding
    local temp_concat="$temp_dir/concat.m4b"
    local ffmpeg_opts=(-v error -stats)
    if [[ "$VERBOSE" == true ]]; then
        ffmpeg_opts=(-v info)
    fi

    # Use -safe 0 but we've validated paths don't contain injection attacks
    if ! ffmpeg "${ffmpeg_opts[@]}" -f concat -safe 0 -i "$filelist" -c copy "$temp_concat"; then
        error "Failed to concatenate audio files" 4
    fi

    # Create FFMETADATA file with combined chapters
    local metadata_file="$temp_dir/metadata.txt"
    echo ";FFMETADATA1" > "$metadata_file"
    echo "title=$(escape_ffmetadata "${source_title:-Audiobook}")" >> "$metadata_file"
    echo "artist=$(escape_ffmetadata "${source_artist:-}")" >> "$metadata_file"
    echo "genre=Audiobook" >> "$metadata_file"
    echo "" >> "$metadata_file"

    # Add chapters with adjusted timestamps
    cumulative_offset=0
    for ((i=0; i<${#INPUT_FILES[@]}; i++)); do
        local file="${INPUT_FILES[$i]}"
        local chapters_json
        chapters_json=$(get_chapters "$file")

        # Process each chapter
        local chapter_array
        chapter_array=$(echo "$chapters_json" | jq -c '.chapters[]')

        while IFS= read -r chapter; do
            local title
            local start_time
            local end_time
            local start_ms
            local end_ms

            title=$(echo "$chapter" | jq -r '.tags.title // "Chapter"')
            # Use start_time/end_time (in seconds) instead of start/end (in timebase units)
            start_time=$(echo "$chapter" | jq -r '.start_time')
            end_time=$(echo "$chapter" | jq -r '.end_time')

            # Convert seconds to milliseconds and adjust by cumulative offset
            start_ms=$(echo "($start_time * 1000 + $cumulative_offset) / 1" | bc)
            end_ms=$(echo "($end_time * 1000 + $cumulative_offset) / 1" | bc)

            echo "[CHAPTER]" >> "$metadata_file"
            echo "TIMEBASE=1/1000" >> "$metadata_file"
            echo "START=$start_ms" >> "$metadata_file"
            echo "END=$end_ms" >> "$metadata_file"
            echo "title=$(escape_ffmetadata "$title")" >> "$metadata_file"
            echo "" >> "$metadata_file"
        done <<< "$chapter_array"

        # Update cumulative offset for next file
        cumulative_offset=$((cumulative_offset + durations_ms[i]))
    done

    success "Concatenated ${#INPUT_FILES[@]} files"
    success "Combined $total_chapters chapters"

    # Apply metadata and create final M4B
    local final_ffmpeg_opts=("${ffmpeg_opts[@]}")
    if [[ "${FORCE:-false}" == true ]]; then
        final_ffmpeg_opts=(-y "${final_ffmpeg_opts[@]}")
    fi

    if ! ffmpeg "${final_ffmpeg_opts[@]}" -i "$temp_concat" -i "$metadata_file" \
        -map_metadata 1 -map_chapters 1 -c copy -movflags +faststart "$OUTPUT_FILE"; then
        error "Failed to apply metadata to output file" 4
    fi

    success "Applied metadata"

    echo

    # Show output information
    local output_size
    output_size=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)

    printf "${GREEN}Output: %s${NC}\n" "$OUTPUT_FILE"
    echo "  Duration: $(format_duration "$cumulative_offset")"
    echo "  Chapters: $total_chapters"
    echo "  Size: $(format_size "$output_size")"

    echo
    success "Merge completed successfully!"
}

# Main script entry point
main() {
    check_dependencies
    parse_args "$@"
    merge_m4b_files
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
