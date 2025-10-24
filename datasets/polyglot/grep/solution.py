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

    for file_name in files:
        with open(file_name, 'r') as f:
            lines = f.readlines()

        file_has_match = False  # Track if this file has any matches (for -l flag)
        file_results = []       # Store results for this file if not using -l

        for line_num, line in enumerate(lines, 1):
            # Remove trailing newline for comparison
            line_content = line.rstrip('\n\r')
            
            # Determine if the line matches the pattern based on flags
            line_matches = False

            if is_match_entire_line:
                # The entire line must match the pattern
                comparison_line = line_content if not is_case_insensitive else line_content.lower()
                line_matches = comparison_line == search_pattern
            else:
                # The pattern can match anywhere in the line
                comparison_line = line_content if not is_case_insensitive else line_content.lower()
                line_matches = search_pattern in comparison_line

            # Apply invert flag
            if is_invert_match:
                line_matches = not line_matches

            if line_matches:
                file_has_match = True
                if not is_output_filenames_only:  # If -l flag is not set, add the line to results
                    output_parts = []
                    
                    if prepend_filename:
                        output_parts.append(f"{file_name}:")
                    
                    if is_prepend_line_numbers:
                        output_parts.append(f"{line_num}:")
                    
                    # Add the original line content (with its original ending)
                    output_parts.append(line)
                    
                    file_results.append("".join(output_parts))

        # After processing all lines in the file
        if is_output_filenames_only:
            if file_has_match:
                results.append(f"{file_name}\n")
        else:
            # Add lines collected for this file (if not using -l flag)
            results.extend(file_results)

    # return results
    return "\n".join(results) + ("\n" if results else "")
