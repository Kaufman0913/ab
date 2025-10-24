def grep(pattern, flags, files):
    """
    Search files for lines matching a search string and return all matching lines.

    Args:
        pattern: The string to search for.
        flags: A list of strings representing flags to customize behavior.
               Supported flags: '-n', '-l', '-i', '-v', '-x'.
        files: A list of strings representing filenames to search in.

    Returns:
        A list of strings representing the output lines based on the search
        and flags.
    """
    results = []
    is_case_insensitive = '-i' in flags
    is_match_entire_line = '-x' in flags
    is_invert_match = '-v' in flags
    is_output_filenames_only = '-l' in flags
    is_prepend_line_numbers = '-n' in flags

    # Prepare pattern for comparison based on case sensitivity
    search_pattern = pattern if not is_case_insensitive else pattern.lower()

    # Check if we need to prepend filenames (when searching multiple files)
    prepend_filename = len(files) > 1

    # Track which files have matches for -l flag, preserving order
    files_with_matches = []

    for file_name in files:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            # Skip files that don't exist
            continue

        # Process lines in the current file
        for line_num, line in enumerate(lines, 1):
            # Remove trailing newline for comparison and processing
            line_content = line.rstrip('\n\r')

            # Determine if the line matches the pattern based on flags
            line_matches = False

            if is_match_entire_line:
                # The entire line content (excluding the terminating <newline>) must match
                comparison_line = line_content if not is_case_insensitive else line_content.lower()
                line_matches = comparison_line == search_pattern
            else:
                # The pattern can match anywhere in the line content (excluding the terminating <newline>)
                comparison_line = line_content if not is_case_insensitive else line_content.lower()
                line_matches = search_pattern in comparison_line

            # Apply invert flag
            if is_invert_match:
                line_matches = not line_matches

            if line_matches:
                # Track file for -l flag if not already tracked
                if file_name not in files_with_matches:
                    files_with_matches.append(file_name)

                # If not using -l, add the matching line to results
                if not is_output_filenames_only:
                    output_parts = []

                    if prepend_filename:
                        output_parts.append(f"{file_name}:")

                    if is_prepend_line_numbers:
                        output_parts.append(f"{line_num}:")

                    # Add the original line content (without its original ending)
                    output_parts.append(line_content)

                    # Add newline to match expected output format for each line
                    output_parts.append("\n")

                    results.append("".join(output_parts))

    # If -l flag is used, return the names of files with matches
    if is_output_filenames_only:
        # Each filename should be followed by a newline
        return [f"{filename}\n" for filename in files_with_matches]

    # Return the list of matching lines (each ending with \n)
    return results