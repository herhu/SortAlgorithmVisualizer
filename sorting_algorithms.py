import matplotlib.pyplot as plt
import numpy as np

def heapify(arr, n, i, bars, sound_manager):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[i] < arr[left]:
        largest = left
    
    if right < n and arr[largest] < arr[right]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        update_bars(arr, bars)
        sound_manager.sound_access(arr[i], arr[largest])
        heapify(arr, n, largest, bars, sound_manager)

def heap_sort(arr, bars, sound_manager):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, bars, sound_manager)
    
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        update_bars(arr, bars)
        sound_manager.sound_access(arr[i], arr[0])
        heapify(arr, i, 0, bars, sound_manager)

def selection_sort(arr, bars, sound_manager):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        update_bars(arr, bars)
        sound_manager.sound_access(arr[i], arr[min_idx])

def insertion_sort(arr, bars, sound_manager):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        update_bars(arr, bars)
        sound_manager.sound_access(arr[j + 1], key)

def shell_sort(arr, bars, sound_manager):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
            update_bars(arr, bars)
            sound_manager.sound_access(arr[j], temp)
        gap //= 2

def quick_sort_dual_pivot(arr, bars, sound_manager):
    def partition(arr, low, high):
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        pivot1, pivot2 = arr[low], arr[high]
        i, j, k, p = low + 1, low + 1, high - 1, low + 1
        while p <= k:
            if arr[p] < pivot1:
                arr[p], arr[i] = arr[i], arr[p]
                i += 1
            elif arr[p] >= pivot2:
                while arr[k] > pivot2 and p < k:
                    k -= 1
                arr[p], arr[k] = arr[k], arr[p]
                k -= 1
                if arr[p] < pivot1:
                    arr[p], arr[i] = arr[i], arr[p]
                    i += 1
            p += 1
        i -= 1
        j -= 1
        arr[low], arr[i] = arr[i], arr[low]
        arr[high], arr[j] = arr[j], arr[high]
        return i, j
    
    def quick_sort(arr, low, high, bars, sound_manager):
        if low < high:
            lp, rp = partition(arr, low, high)
            quick_sort(arr, low, lp - 1, bars, sound_manager)
            quick_sort(arr, lp + 1, rp - 1, bars, sound_manager)
            quick_sort(arr, rp + 1, high, bars, sound_manager)
            update_bars(arr, bars)
    
    quick_sort(arr, 0, len(arr) - 1, bars, sound_manager)

# 4. Quick Sort (ternary, LR ptrs)
def quick_sort_ternary_lr(arr, bars, sound_manager):
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                sound_manager.sound_access(arr[i], arr[j])
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        sound_manager.sound_access(arr[i + 1], arr[high])
        return i + 1
    
    def quick_sort(arr, low, high, bars, sound_manager):
        if low < high:
            pi = partition(arr, low, high)
            quick_sort(arr, low, pi - 1, bars, sound_manager)
            quick_sort(arr, pi + 1, high, bars, sound_manager)
            update_bars(arr, bars)
    
    quick_sort(arr, 0, len(arr) - 1, bars, sound_manager)

# 5. Quick Sort (LR ptrs)
def quick_sort_lr(arr, bars, sound_manager):
    def partition(arr, low, high):
        pivot = arr[low]
        left = low + 1
        right = high
        while True:
            while left <= right and arr[left] <= pivot:
                left += 1
            while left <= right and arr[right] >= pivot:
                right -= 1
            if left <= right:
                arr[left], arr[right] = arr[right], arr[left]
                sound_manager.sound_access(arr[left], arr[right])
            else:
                break
        arr[low], arr[right] = arr[right], arr[low]
        sound_manager.sound_access(arr[low], arr[right])
        return right
    
    def quick_sort(arr, low, high, bars, sound_manager):
        if low < high:
            pi = partition(arr, low, high)
            quick_sort(arr, low, pi - 1, bars, sound_manager)
            quick_sort(arr, pi + 1, high, bars, sound_manager)
            update_bars(arr, bars)
    
    quick_sort(arr, 0, len(arr) - 1, bars, sound_manager)

# 6. Quick Sort (LL ptrs)
def quick_sort_ll(arr, bars, sound_manager):
    def partition(arr, low, high):
        pivot = arr[low]
        left = low + 1
        right = high
        while left <= right:
            while left <= right and arr[left] <= pivot:
                left += 1
            while left <= right and arr[right] > pivot:
                right -= 1
            if left < right:
                arr[left], arr[right] = arr[right], arr[left]
                sound_manager.sound_access(arr[left], arr[right])
        arr[low], arr[right] = arr[right], arr[low]
        sound_manager.sound_access(arr[low], arr[right])
        return right
    
    def quick_sort(arr, low, high, bars, sound_manager):
        if low < high:
            pi = partition(arr, low, high)
            quick_sort(arr, low, pi - 1, bars, sound_manager)
            quick_sort(arr, pi + 1, high, bars, sound_manager)
            update_bars(arr, bars)
    
    quick_sort(arr, 0, len(arr) - 1, bars, sound_manager)

# 7. Quick Sort (ternary, LL ptrs)
def quick_sort_ternary_ll(arr, bars, sound_manager):
    def partition(arr, low, high):
        pivot = arr[low]
        left = low + 1
        right = high
        while left <= right:
            while left <= right and arr[left] <= pivot:
                left += 1
            while left <= right and arr[right] > pivot:
                right -= 1
            if left < right:
                arr[left], arr[right] = arr[right], arr[left]
                sound_manager.sound_access(arr[left], arr[right])
        arr[low], arr[right] = arr[right], arr[low]
        sound_manager.sound_access(arr[low], arr[right])
        return right
    
    def quick_sort(arr, low, high, bars, sound_manager):
        if low < high:
            pi = partition(arr, low, high)
            quick_sort(arr, low, pi - 1, bars, sound_manager)
            quick_sort(arr, pi + 1, high, bars, sound_manager)
            update_bars(arr, bars)
    
    quick_sort(arr, 0, len(arr) - 1, bars, sound_manager)

# 8. Tim Sort
def tim_sort(arr, bars, sound_manager):
    def insertion_sort(arr, left, right):
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
            sound_manager.sound_access(arr[j + 1], key)

    def merge(arr, left, mid, right):
        len1, len2 = mid - left + 1, right - mid
        left_arr, right_arr = arr[left:mid + 1], arr[mid + 1:right + 1]

        i, j, k = 0, 0, left

        while i < len1 and j < len2:
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            k += 1

        while i < len1:
            arr[k] = left_arr[i]
            i += 1
            k += 1

        while j < len2:
            arr[k] = right_arr[j]
            j += 1
            k += 1

    def tim_sort_util(arr, n, bars, sound_manager):
        min_run = 32
        for i in range(0, n, min_run):
            insertion_sort(arr, i, min((i + min_run - 1), n - 1))
        
        size = min_run
        while size < n:
            for left in range(0, n, 2 * size):
                mid = left + size - 1
                right = min((left + 2 * size - 1), (n - 1))
                merge(arr, left, mid, right)
                sound_manager.sound_access(arr[left], arr[right])
            size *= 2

    tim_sort_util(arr, len(arr), bars, sound_manager)

# 10. std::sort (gcc)
def std_sort(arr, bars, sound_manager):
    arr.sort()
    update_bars(arr, bars)

# 11. Merge Sort
def merge_sort(arr, bars, sound_manager):
    def merge(arr, l, m, r):
        n1 = m - l + 1
        n2 = r - m
        L = arr[l:l + n1].copy()
        R = arr[m + 1:r + 1].copy()
        i = j = 0
        k = l
        while i < n1 and j < n2:
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < n1:
            arr[k] = L[i]
            i += 1
            k += 1
        while j < n2:
            arr[k] = R[j]
            j += 1
            k += 1
        sound_manager.sound_access(arr[l], arr[r])
        update_bars(arr, bars)

    def merge_sort_util(arr, l, r):
        if l < r:
            m = (l + r) // 2
            merge_sort_util(arr, l, m)
            merge_sort_util(arr, m + 1, r)
            merge(arr, l, m, r)

    merge_sort_util(arr, 0, len(arr) - 1)

# 12. Smooth Sort
def smooth_sort(arr, bars, sound_manager):
    def sift_down(arr, l, r, sound_manager):
        n = r - l
        if n <= 1:
            return
        m = n // 2
        while m >= 1:
            k = m
            while k <= n // 2:
                child = 2 * k
                if child + 1 <= n and arr[l + child - 1] < arr[l + child]:
                    child += 1
                if arr[l + k - 1] >= arr[l + child - 1]:
                    break
                arr[l + k - 1], arr[l + child - 1] = arr[l + child - 1], arr[l + k - 1]
                sound_manager.sound_access(arr[l + k - 1], arr[l + child - 1])
                update_bars(arr, bars)
                k = child
            m //= 2

    def smooth_sort_util(arr, l, r):
        if l >= r:
            return
        m = (l + r) // 2
        smooth_sort_util(arr, l, m)
        smooth_sort_util(arr, m + 1, r)
        sift_down(arr, l, r, sound_manager)

    smooth_sort_util(arr, 0, len(arr) - 1)

# 13. Block Merge Sort
def block_merge_sort(arr, bars, sound_manager):
    def merge_blocks(arr, l, m, r):
        len1, len2 = m - l + 1, r - m
        left, right = arr[l:l + len1].copy(), arr[m + 1:m + 1 + len2].copy()
        i, j, k = 0, 0, l
        while i < len1 and j < len2:
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len1:
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len2:
            arr[k] = right[j]
            j += 1
            k += 1
        sound_manager.sound_access(arr[l], arr[r])
        update_bars(arr, bars)

    def block_merge_sort_util(arr, l, r):
        block_size = 32
        for i in range(l, r + 1, block_size):
            insertion_sort(arr, i, min(i + block_size - 1, r))
        size = block_size
        while size < r - l + 1:
            for left in range(l, r + 1, 2 * size):
                mid = left + size - 1
                right = min(left + 2 * size - 1, r)
                merge_blocks(arr, left, mid, right)
            size *= 2

    block_merge_sort_util(arr, 0, len(arr) - 1)

# 14. Comb Sort
def comb_sort(arr, bars, sound_manager):
    def getNextGap(gap):
        gap = (gap * 10) // 13
        if gap < 1:
            return 1
        return gap

    n = len(arr)
    gap = n
    swapped = True
    while gap != 1 or swapped:
        gap = getNextGap(gap)
        swapped = False
        for i in range(0, n - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sound_manager.sound_access(arr[i], arr[i + gap])
                update_bars(arr, bars)
                swapped = True

# 15. Bitonic Sort
def bitonic_sort(arr, bars, sound_manager):
    def bitonic_merge(arr, low, cnt, dir, bars, sound_manager):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if (arr[i] > arr[i + k]) == dir:
                    arr[i], arr[i + k] = arr[i + k], arr[i]
                    sound_manager.sound_access(arr[i], arr[i + k])
                    update_bars(arr, bars)
            bitonic_merge(arr, low, k, dir, bars, sound_manager)
            bitonic_merge(arr, low + k, k, dir, bars, sound_manager)

    def bitonic_sort_util(arr, low, cnt, dir, bars, sound_manager):
        if cnt > 1:
            k = cnt // 2
            bitonic_sort_util(arr, low, k, 1, bars, sound_manager)
            bitonic_sort_util(arr, low + k, k, 0, bars, sound_manager)
            bitonic_merge(arr, low, cnt, dir, bars, sound_manager)

    bitonic_sort_util(arr, 0, len(arr), 1, bars, sound_manager)

# 17. Binary Insertion Sort
def binary_insertion_sort(arr, bars, sound_manager):
    def binary_search(arr, item, low, high):
        if high <= low:
            return (low + 1) if item > arr[low] else low
        mid = (low + high) // 2
        if item == arr[mid]:
            return mid + 1
        elif item > arr[mid]:
            return binary_search(arr, item, mid + 1, high)
        else:
            return binary_search(arr, item, low, mid - 1)

    n = len(arr)
    for i in range(1, n):
        j = i - 1
        key = arr[i]
        loc = binary_search(arr, key, 0, j)
        while j >= loc:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        update_bars(arr, bars)
        sound_manager.sound_access(arr[j + 1], key)

# 20. Cocktail Shaker Sort
def cocktail_shaker_sort(arr, bars, sound_manager):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1
    while swapped:
        swapped = False
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sound_manager.sound_access(arr[i], arr[i + 1])
                update_bars(arr, bars)
                swapped = True
        if not swapped:
            break
        swapped = False
        end -= 1
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sound_manager.sound_access(arr[i], arr[i + 1])
                update_bars(arr, bars)
                swapped = True
        start += 1

# 21. Gnome Sort
def gnome_sort(arr, bars, sound_manager):
    n = len(arr)
    index = 0
    while index < n:
        if index == 0:
            index += 1
        if arr[index] >= arr[index - 1]:
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            sound_manager.sound_access(arr[index], arr[index - 1])
            update_bars(arr, bars)
            index -= 1

# 22. Cycle Sort
def cycle_sort(arr, bars, sound_manager):
    n = len(arr)
    for cycle_start in range(0, n - 1):
        item = arr[cycle_start]
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            if arr[i] < item:
                pos += 1
        if pos == cycle_start:
            continue
        while item == arr[pos]:
            pos += 1
        arr[pos], item = item, arr[pos]
        sound_manager.sound_access(arr[pos], item)
        update_bars(arr, bars)
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1
            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]
            sound_manager.sound_access(arr[pos], item)
            update_bars(arr, bars)

# 23. Bubble Sort
def bubble_sort(arr, bars, sound_manager):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                sound_manager.sound_access(arr[j], arr[j + 1])
                update_bars(arr, bars)
                swapped = True
        if not swapped:
            break

# 24. Odd-Even Sort
def odd_even_sort(arr, bars, sound_manager):
    n = len(arr)
    sorted = False
    while not sorted:
        sorted = True
        for i in range(1, n - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sound_manager.sound_access(arr[i], arr[i + 1])
                update_bars(arr, bars)
                sorted = False
        for i in range(0, n - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sound_manager.sound_access(arr[i], arr[i + 1])
                update_bars(arr, bars)
                sorted = False

# Radix Sort (LSD)
def radix_sort_lsd(arr, bars, sound_manager):
    def counting_sort(arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
        
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
        
        for i in range(n):
            arr[i] = output[i]
            sound_manager.sound_access(arr[i])

    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort(arr, exp)
        update_bars(arr, bars)
        exp *= 10

# Radix Sort (MSD)
def radix_sort_msd(arr, bars, sound_manager):
    def radix_sort_recursive(arr, start, end, exp):
        if start >= end or exp <= 0:
            return
        
        count = [0] * 10
        output = [0] * (end - start + 1)
        
        for i in range(start, end + 1):
            index = (arr[i] // exp) % 10
            count[index] += 1
        
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        i = end
        while i >= start:
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            i -= 1
        
        for i in range(start, end + 1):
            arr[i] = output[i - start]
            sound_manager.sound_access(arr[i])
        
        update_bars(arr, bars)
        
        for i in range(10):
            radix_sort_recursive(arr, start + count[i - 1], start + count[i] - 1, exp // 10)

    max_val = max(arr)
    radix_sort_recursive(arr, 0, len(arr) - 1, max_val)

# Shell Sort (Hibbard's increments)
def shell_sort_hibbard(arr, bars, sound_manager):
    n = len(arr)
    gap = 1
    while gap < n / 3:
        gap = 3 * gap + 1
    
    while gap >= 1:
        for i in range(gap, n):
            j = i
            while j >= gap and arr[j] < arr[j - gap]:
                arr[j], arr[j - gap] = arr[j - gap], arr[j]
                sound_manager.sound_access(arr[j], arr[j - gap])
                update_bars(arr, bars)
                j -= gap
        gap //= 3

# Helper function to update bars in the visualization
def update_bars(arr, bars):
    for bar, val in zip(bars, arr):
        bar.set_height(val)
    plt.pause(0.001)

# End of sorting_algorithms.py
