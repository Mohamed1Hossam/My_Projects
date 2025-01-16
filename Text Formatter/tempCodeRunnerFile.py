    if output_text_box.edit_modified():  # Check if the text has been modified
        # Get the position of the cursor
        cursor_position = output_text_box.index(tk.INSERT)
        
        # Determine the start and end index of the last character typed
        start_index = f"{cursor_position}-1c"  # Start one character before the cursor
        end_index = cursor_position  # End at the cursor position
        
        # Remove previous green tags in the last character area
        output_text_box.tag_remove("green_tag", start_index, end_index)
        
        # Add green tag to the last character typed
        output_text_box.tag_add("green_tag", start_index, end_index)
        
        output_text_box.edit_modified(False)  # Reset the modified flag

    # Configure green color for the specific text added
    output_text_box.tag_config("green_tag", foreground="green")