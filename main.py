import matplotlib.pyplot as plt
import numpy as np
from sound import SoundManager
from sorting_algorithms import *

def animate_sorting(arr, algorithm):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')  # Set background to black
    ax.set_title(f"{algorithm.__name__.replace('_', ' ').title()} Visualization with Sound", color='red')
    bars = ax.bar(range(len(arr)), arr, align="edge", color='blue')
    ax.set_xlim(0, len(arr))
    ax.set_ylim(0, int(1.1 * max(arr)))
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='orange')
    plt.ion()
    plt.show()
    
    # Initialize sound manager
    sound_manager = SoundManager(arr)
    
    # Start sorting algorithm with sound
    algorithm(arr, bars, sound_manager)
    
    # Keep the plot window open until user closes it
    plt.ioff()
    plt.show()

def print_menu():
    print("Choose a sorting algorithm to visualize:")
    print("1. Heap Sort")
    print("2. Selection Sort")
    print("3. Shell Sort")
    print("4. Insertion Sort")
    print("5. Radix Sort (LSD)")
    print("6. Radix Sort (MSD)")
    print("7. Quick Sort (dual pivot)")
    print("8. Quick Sort (ternary, LR ptrs)")
    print("9. Quick Sort (LR ptrs)")
    print("10. Quick Sort (LL ptrs)")
    print("11. Quick Sort (ternary, LL ptrs)")
    print("12. Tim Sort")
    print("13. Shell Sort (Hibbard's increments)")
    print("14. std::sort (gcc)")
    print("15. Merge Sort")
    print("16. Smooth Sort")
    print("17. Block Merge Sort")
    print("18. Comb Sort")
    print("19. Bitonic Sort")
    print("20. Binary Insertion Sort")
    print("21. Cocktail Shaker Sort")
    print("22. Gnome Sort")
    print("23. Cycle Sort")
    print("24. Bubble Sort")
    print("25. Odd-Even Sort")

def main():
    # Example array
    arr = np.random.randint(1, 100, 50)
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-25, 0 to quit): ")
        
        if choice == '0':
            print("Exiting...")
            break
        
        try:
            choice = int(choice)
            if choice < 1 or choice > 25:
                print("Invalid choice. Please enter a number between 1 and 25.")
                continue
            
            algorithms = [
                heap_sort, selection_sort, shell_sort, insertion_sort,
                radix_sort_lsd, radix_sort_msd, quick_sort_dual_pivot,
                quick_sort_ternary_lr, quick_sort_lr, quick_sort_ll,
                quick_sort_ternary_ll, tim_sort, shell_sort_hibbard,
                std_sort, merge_sort, smooth_sort, block_merge_sort,
                comb_sort, bitonic_sort, binary_insertion_sort,
                cocktail_shaker_sort, gnome_sort, cycle_sort,
                bubble_sort, odd_even_sort
            ]
            
            animate_sorting(arr.copy(), algorithms[choice - 1])
        
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue

if __name__ == "__main__":
    main()
