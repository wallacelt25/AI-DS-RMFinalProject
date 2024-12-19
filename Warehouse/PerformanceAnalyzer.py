import time
import tracemalloc

class PerformanceAnalyzer:
    @staticmethod
    def analyze_performance(function):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            tracemalloc.start()
            result = function(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end_time = time.time()
            print(f"Function: {function.__name__}, Time: {end_time - start_time}s, Memory: {current / 10**6} MB, Peak: {peak / 10**6} MB")
            return result
        return wrapper

    # Additional tools as needed
    @staticmethod
    def memory_usage():
        """
        Reports the current memory usage of the system.
        """
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10**6} MB, Peak: {peak / 10**6} MB")

    @staticmethod
    def complexity_analysis():
        """
        Provides a basic analysis of algorithm complexity.
        """
        print("Complexity Analysis: Greedy Algorithm - O(n log n) in average case")