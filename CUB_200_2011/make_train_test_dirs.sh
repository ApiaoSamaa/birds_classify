#!/bin/zsh

# The main directory containing all category directories
ORIGINAL_DIR='/Users/a123/proj/FGVC-HERBS/CUB_200_2011/CUB_200_2011/images'
# The directory where you want to create symbolic links to all subcategories
TRAIN_DIR="/Users/a123/proj/FGVC-HERBS/CUB_200_2011/train"
TEST_DIR="/Users/a123/proj/FGVC-HERBS/CUB_200_2011/test"

# Create the TRAIN_DIR and TEST_DIR if they don't exist
mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

# Loop through each category and subcategory to create symbolic links
for category in $(ls "$ORIGINAL_DIR"); do
    mkdir -p "$TRAIN_DIR/$category"
    mkdir -p "$TEST_DIR/$category"
    COUNTER=0  # Reset COUNTER for each new category
    for pic in $(ls "$ORIGINAL_DIR/$category"); do
        if ((COUNTER > 35)); then
            # Create a symbolic link to the subcategory in the TEST_DIR
            ln -s "$ORIGINAL_DIR/$category/$pic" "$TEST_DIR/$category/$pic"
            printf "TRAIN_DIR/$category/$pic\n"
        fi
        if ((COUNTER <= 35)); then
            # Create a symbolic link to the subcategory in the TRAIN_DIR
            ln -s "$ORIGINAL_DIR/$category/$pic" "$TRAIN_DIR/$category/$pic"
            printf "TEST_DIR/$category/$pic\n"
        fi
        ((COUNTER++))  # Increment COUNTER after creating the symbolic link
        echo $COUNTER
    done
done
