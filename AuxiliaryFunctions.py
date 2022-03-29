# expects sorted array and executes binary search in the subarray
# v[start:end] searching for elem.
# Return the index of the element if found, otherwise returns -1
def BinarySearch(v, elem, start=0, end=None):
    if end == None:
        end = len(v)
    
    while start < end:
        mid = int((start + end)/2)
        if elem <= v[mid]:
            end = mid
        else:
            start = mid + 1

    return end if v[end] == elem else -1
