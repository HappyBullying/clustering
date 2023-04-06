#pragma once

#include <cmath>
#include <random>
#include <omp.h>
#include <mutex>
#include <float.h>
#include <string.h>
#include "metrics.h"

using namespace std;

namespace clustering
{
	// void (*means_sum_function)(double *target, const double *source, const size_t n)
	inline void calculate_centroids(double** data, double** means, double** old_means, double** helper_means, size_t* labels, size_t points_count, size_t number_of_clusters, size_t points_dim, size_t* counts_helper)
	{
		for (size_t point = 0; point < points_count; ++point)
		{
			size_t cluster = labels[point];
			double* curr_helper_mean_point = helper_means[cluster];
			double* curr_point = data[point];
			for (size_t j = 0; j < points_dim; j++)
			{
				curr_helper_mean_point[j] += curr_point[j];
			}
			counts_helper[cluster] += 1;
		}

		// Divide sums by counts to get new centroids.
		for (size_t cluster = 0; cluster < number_of_clusters; ++cluster)
		{
			// Turn 0/0 into 0/1 to avoid zero division.
			size_t count = std::max<size_t>(1, counts_helper[cluster]);
			double count_inv = 1.0 / ((double)count);

			double* curr_mean_point = means[cluster];
			double* curr_helper_mean_point = helper_means[cluster];
			double* current_old_means = old_means[cluster];
			memcpy(current_old_means, curr_mean_point, sizeof(double) * points_dim);

			for (size_t j = 0; j < points_dim; j++)
			{
				curr_mean_point[j] = curr_helper_mean_point[j] * count_inv;
			}
		}
	}

	typedef double (*kmeans_sum_of_squared_deviations_func)(double* x1, double* x2, double* weights, size_t n);
	typedef void (*kmeans_calculate_centroids_func)(double** data, double** means, double** old_means, double** helper_means, size_t* labels, size_t points_count, size_t number_of_clusters, size_t points_dim, size_t* counts_helper);

	class KMeans
	{
	public:

		KMeans(size_t number_of_clusters, size_t n_init) {
			this->points = nullptr;
			this->centroids = nullptr;
			this->number_of_clusters = number_of_clusters;
			this->n_init = n_init;
			this->relative_decrease = -0.0001;
			this->ssdf = metrics::sum_of_squared_deviations_weightless;
			this->ccf = clustering::calculate_centroids;
			this->labels = nullptr;
			this->centroids = nullptr;
		}

		KMeans(size_t number_of_clusters, size_t n_init, double relative_decrease, kmeans_sum_of_squared_deviations_func ssdf, kmeans_calculate_centroids_func ccf)
		{
			this->points = nullptr;
			this->centroids = nullptr;
			this->number_of_clusters = number_of_clusters;
			this->n_init = n_init;
			this->relative_decrease = relative_decrease;
			this->ssdf = ssdf;
			this->ccf = ccf;
			this->labels = nullptr;
			this->centroids = nullptr;
		}


		KMeans(size_t number_of_clusters, double** centroids, size_t points_dim, double relative_decrease, kmeans_sum_of_squared_deviations_func ssdf, kmeans_calculate_centroids_func ccf)
		{
			this->number_of_clusters = number_of_clusters;
			this->n_init = 1;
			this->relative_decrease = relative_decrease;
			this->ssdf = ssdf;
			this->ccf = ccf;
			this->labels = nullptr;
			this->centroids = (double**)malloc(sizeof(double*) * number_of_clusters);

			size_t centroid_size = sizeof(double) * points_dim;
			for (size_t i = 0; i < number_of_clusters; i++) {
				this->centroids[i] = (double*)malloc(centroid_size);
				memcpy(this->centroids[i], centroids[i], centroid_size);
			}
			this->default_centroids_provided = true;
		}


		void run(double** data, double* sample_weights, size_t points_count, size_t points_dim) {
			this->run(data, sample_weights, points_count, points_dim, 25, 600);
		}


		void run(double** data, double* sample_weights, size_t points_count, size_t points_dim, size_t min_number_of_iterations, size_t max_number_of_iterations) {
			this->points = data;
			this->points_count = points_count;
			this->points_dim = points_dim;

			this->initialize_random_device(this);

			if (sample_weights != nullptr) {
				size_t weights_size = sizeof(double) * this->points_dim;
				this->weights = (double*)malloc(weights_size);
				memcpy(this->weights, sample_weights, weights_size);
			}

			size_t** all_n_labels = (size_t**)malloc(sizeof(size_t*) * this->n_init);
			size_t* iterations_arr = (size_t*)malloc(sizeof(size_t) * this->n_init);
			double* inertia_array = (double*)malloc(sizeof(double) * this->n_init);
			double*** all_n_means = (double***)malloc(sizeof(double**) * this->n_init);
			double*** all_helper_means = (double***)malloc(sizeof(double**) * this->n_init);
			double*** all_old_means = (double***)malloc(sizeof(double**) * this->n_init);
			

			/*for (size_t i = 0; i < this->n_init; i++) {
				all_helper_means[i] = (double**)malloc(sizeof(double*) * this->number_of_clusters);
				all_old_means[i] = (double**)malloc(sizeof(double*) * this->number_of_clusters);
				for (size_t j = 0; j < this->number_of_clusters; j++) {
					all_helper_means[i][j] = (double*)malloc(points_dim * sizeof(double));
					all_old_means[i][j] = (double*)malloc(points_dim * sizeof(double));
				}
			}*/

//			omp_set_dynamic(0);     // Explicitly disable dynamic teams
//			omp_set_num_threads(4)
//#pragma omp parallel for num_threads(variable)

			// !!!!!!!!! OMP REPLACE HERE ALL THIS WITH LOCAL COPY OF THIS

			for (size_t current_n = 0; current_n < this->n_init; current_n++) {
				if (this->default_centroids_provided) { all_n_means[current_n] = this->centroids; }
				else { all_n_means[current_n] = initialize_centroids(this); }
				all_helper_means[current_n] = (double**)malloc(sizeof(double*) * number_of_clusters);
				all_old_means[current_n] = (double**)malloc(sizeof(double*) * number_of_clusters);

				for (size_t j = 0; j < this->number_of_clusters; j++) {
					all_helper_means[current_n][j] = (double*)malloc(sizeof(double) * this->points_dim);
					all_old_means[current_n][j] = (double*)malloc(sizeof(double) * this->points_dim);
				}

				size_t* counts_helper = (size_t*)malloc(sizeof(size_t) * this->number_of_clusters);
				size_t* labels = (size_t*)calloc(this->points_count, sizeof(size_t));
				all_n_labels[current_n] = labels;

				size_t iteration = 0;
				double max_ssd_delta_prev = DBL_MAX;
				double relative_ssd_delta_prev = DBL_MAX;
				bool continue_converging = true;
				
				for (; (continue_converging && (iteration < max_number_of_iterations)) || (iteration < min_number_of_iterations); ++iteration)
				{
					// Find assignments.
					find_assignments(labels, all_n_means[current_n]);

					// Reset helper_means and counts
					for (size_t i = 0; i < number_of_clusters; i++) {
						std::fill_n(all_helper_means[current_n][i], this->points_dim, 0.0);
					}
					std::fill_n(counts_helper, number_of_clusters, 0);

					// Sum up and count points for each cluster.
					this->ccf(this->points, all_n_means[current_n], all_old_means[current_n], all_helper_means[current_n], labels, this->points_count, this->number_of_clusters, this->points_dim, counts_helper);

					continue_converging = should_continue(all_n_means[current_n], all_old_means[current_n], max_ssd_delta_prev, relative_ssd_delta_prev);
				}
				iterations_arr[current_n] = iteration;
				calculate_inertia_for_n(all_n_means[current_n], inertia_array, labels, current_n);
				free(counts_helper);
			}


#pragma region FIND_INIT_WITH_LOWEST_INERTIA	
			double min_inertia;
			size_t best_init_index;
			size_t total_iterations;
			find_case_with_least_inertia(inertia_array, iterations_arr, min_inertia, best_init_index, total_iterations);
#pragma endregion


			this->inertia = min_inertia;
			this->labels = all_n_labels[best_init_index];
			this->centroids = all_n_means[best_init_index];
			this->total_iterations = total_iterations;


#pragma region RELEASE_MEMORY			
			for (size_t i = 0; i < this->n_init; i++) {
				if (i != best_init_index) {
					for (size_t j = 0; j < this->number_of_clusters; j++) {
						free(all_helper_means[i][j]);
						free(all_old_means[i][j]);
						free(all_n_means[i][j]);
					}
					free(all_helper_means[i]);
					free(all_old_means[i]);
					free(all_n_means[i]);
					free(all_n_labels[i]);
				}
			}

			free(all_helper_means);
			free(all_old_means);
			free(all_n_means);
			free(all_n_labels);

			free(inertia_array);
			free(iterations_arr);
#pragma endregion
		}


		~KMeans() {
			if (labels != nullptr) {
				free(labels);
			}

			if (centroids != nullptr) {
				for (size_t i = 0; i < number_of_clusters; i++) {
					if (centroids[i] != nullptr) {
						free(centroids[i]);
					}
				}
				free(centroids);
			}

			if (this->random_devices_initialized) {
				delete this->indices;
				delete this->random_number_generator;
			}
		}

		

		size_t get_point_dim() {
			return this->points_dim;
		}

		size_t get_number_of_clusters() {
			return this->number_of_clusters;
		}

		double get_inertia() {
			return this->inertia;
		}

		size_t get_iterations() {
			return this->total_iterations;
		}


		size_t* get_labels() {
			return this->labels;
		}

		size_t* get_labels_copy() {
			size_t labels_size = sizeof(size_t) * this->number_of_clusters;
			size_t* labels = (size_t*)malloc(labels_size);
			memcpy(labels, this->labels, labels_size);

			return labels;
		}


		double** get_centroids() {
			return this->centroids;
		}

		double** get_centroids_copy() {
			double** centroids = (double**)malloc(sizeof(double*) * this->number_of_clusters);
			size_t centroid_size = sizeof(double)* this->points_dim;

			for (size_t i = 0; i < this->number_of_clusters; i++) {
				centroids[i] = (double*)malloc(centroid_size);
				memcpy(centroids[i], this->centroids[i], centroid_size);
			}

			return centroids;
		}


	private:

		static inline void initialize_random_device(KMeans* instance) {
			if (instance->random_devices_initialized)
				return;

			std::random_device random_seed;
			instance->random_number_generator = new std::mt19937(random_seed());
			instance->indices = new std::uniform_int_distribution<size_t>(0, instance->points_count - 1);
			instance->random_devices_initialized = true;
		}


		static size_t random_szie_t(KMeans* instance) {
			std::unique_lock<std::recursive_mutex> lock(instance->m);
			size_t rnd = instance->random_number_generator->operator()();
			lock.unlock();
			return rnd;
		}


		static size_t random_point_index(KMeans* instance) {
			std::unique_lock<std::recursive_mutex> lock(instance->m);
			size_t rnd = instance->indices->operator()(*(instance->random_number_generator));
			lock.unlock();
			return rnd;
		}


		static double random_double(KMeans* instance) {
			std::unique_lock<std::recursive_mutex> lock(instance->m);
			double rnd_double = (double)(instance->random_szie_t(instance));
			rnd_double = rnd_double / (instance->random_number_generator->max() - instance->random_number_generator->min());
			lock.unlock();
			return rnd_double;
		}


		inline void find_assignments(size_t* labels, double** means)
		{
			for (size_t point_ind = 0; point_ind < points_count; ++point_ind)
			{
				double best_sum_of_squared_deviations = DBL_MAX;
				size_t best_cluster = 0;
				for (size_t cluster = 0; cluster < number_of_clusters; ++cluster)
				{
					double sum_of_squared_deviations = ssdf(points[point_ind], means[cluster], weights, points_dim);
					if (sum_of_squared_deviations < best_sum_of_squared_deviations)
					{
						best_sum_of_squared_deviations = sum_of_squared_deviations;
						best_cluster = cluster;
					}
				}
				labels[point_ind] = best_cluster;
			}
		}





		inline bool should_continue(double** means, double** old_means, double& max_ssd_delta_prev, double& relative_ssd_delta_prev) {
			double max_ssd_delta_current = 0;
			for (size_t cl = 0; cl < number_of_clusters; cl++) {
				double ssd_delta_tmp = ssdf(means[cl], old_means[cl], weights, points_dim); // ??????
				if (ssd_delta_tmp > max_ssd_delta_current) {
					max_ssd_delta_current = ssd_delta_tmp;
				}
			}
			double relative_ssd_delta = (max_ssd_delta_current - max_ssd_delta_prev) / (max_ssd_delta_prev == 0 ? 1 : max_ssd_delta_prev);
			max_ssd_delta_prev = max_ssd_delta_current;

			bool result = false;
			if (relative_ssd_delta <= 0 && relative_ssd_delta >= relative_decrease && relative_ssd_delta_prev <= 0 && relative_ssd_delta_prev >= relative_decrease && relative_ssd_delta_prev <= relative_ssd_delta) { result = false; }
			else { result = true; }
			relative_ssd_delta_prev = relative_ssd_delta;
			return result;
		}


		inline void calculate_inertia_for_n(double** means, double* inertia_array, size_t* labels, size_t current_n) {
			double current_inertia = 0;
			for (size_t pt_index = 0; pt_index < points_count; pt_index++) {
				double* curr_pt = points[pt_index];
				double* curr_centroid = means[labels[pt_index]];
				double ssd = ssdf(curr_pt, curr_centroid, weights, points_dim);
				current_inertia += ssd;
			}

			inertia_array[current_n] = current_inertia;
		}


		inline void find_case_with_least_inertia(double* inertia_array, size_t* iterations_arr, double& min_inertia, size_t& best_init_index, size_t& total_iterations) {
			min_inertia = inertia_array[0];
			best_init_index = 0;
			total_iterations = iterations_arr[0];
			for (size_t i = 1; i < n_init; i++) {
				double curr_inertia = inertia_array[i];
				if (curr_inertia < min_inertia) {
					min_inertia = curr_inertia;
					best_init_index = i;
					total_iterations = iterations_arr[i];
				}
			}
		}


		inline double** initialize_centroids(KMeans* instance) {
			double* ssd_arr = (double*)malloc(sizeof(double) * points_count);
			double** temp_centroids = (double**)malloc(sizeof(double*) * number_of_clusters);

			size_t counter = 0;
			for (size_t i = 0; i < number_of_clusters; i++) {
				temp_centroids[i] = (double*)malloc(sizeof(double) * points_dim);
			}

			memcpy(temp_centroids[counter], points[random_point_index(instance)], sizeof(double) * points_dim);
			counter++;


			while (counter < number_of_clusters) {
				double sum = 0;
				for (size_t i = 0; i < points_count; i++) {
					double ssd_best = DBL_MAX;
					for (size_t j = 0; j < counter; j++) {
						double* point_at_i = points[i];
						double* centroid_at_j = temp_centroids[j];
						double ssd = ssdf(point_at_i, centroid_at_j, weights, points_dim);
						if (ssd < ssd_best) {
							ssd_best = ssd;
						}
					}
					ssd_arr[i] = ssd_best;
					sum += ssd_best;
				}

				// Calculating probabilities and cumulative sum of probabilities
				double sum_inv = 1.0 / (sum == 0 ? 1.0 : sum);
				double r = random_double(instance);
				double cum_sum_prob = 0;
				size_t c_index = 0;
				ssd_arr[0] *= sum_inv;
				cum_sum_prob = ssd_arr[0];

				if (cum_sum_prob < r) {
					for (size_t i = 1; i < points_count; i++) {
						ssd_arr[i] *= sum_inv;
						cum_sum_prob += ssd_arr[i];
						if (cum_sum_prob >= r) {
							c_index = i;
							break;
						}
					}
				}

				memcpy(temp_centroids[counter], points[c_index], sizeof(double) * points_dim);
				counter++;
			}

			free(ssd_arr);

			return temp_centroids;
		}

		bool random_devices_initialized = false;
		bool default_centroids_provided = false;
		double** points = nullptr;
		double** centroids = nullptr;
		double* weights = nullptr;
		size_t* labels = nullptr;
		size_t points_count;

		size_t points_dim;
		size_t number_of_clusters;
		size_t n_init;
		size_t total_iterations;
		double inertia;
		double relative_decrease;
		kmeans_sum_of_squared_deviations_func ssdf;
		kmeans_calculate_centroids_func ccf;
		
		std::mt19937* random_number_generator = nullptr;
		std::uniform_int_distribution<size_t>* indices = nullptr;
		std::recursive_mutex m;
		//void (*__helper_means_summer_function)(double *, const double *, const size_t n);
		//void (*__means_assign_function)(double *, const double *, const size_t k, const size_t n);
	};




	inline void calculate_lined_centroids(double** data, size_t* labels, size_t points_count, size_t number_of_clusters, size_t points_dim, double** means, double** old_means, double** helper_means, size_t* counts_helper)
	{
		size_t arr_size = sizeof(double) * number_of_clusters;
		size_t* samples_counts = (size_t*)malloc(sizeof(size_t) * number_of_clusters);
		double* sum_xy = (double*)malloc(arr_size);
		double* sum_x = (double*)malloc(arr_size);
		double* sum_y = (double*)malloc(arr_size);
		double* sum_x_sqr = (double*)malloc(arr_size);
		double* sum_y_sqr = (double*)malloc(arr_size);

		for (size_t cluster = 0; cluster < number_of_clusters; ++cluster)
		{
			sum_xy[cluster] = 0.0;
			sum_x[cluster] = 0.0;
			sum_y[cluster] = 0.0;
			sum_x_sqr[cluster] = 0.0;
			sum_y_sqr[cluster] = 0.0;
			samples_counts[cluster] = 0;
			memcpy(old_means[cluster], means[cluster], sizeof(double) * points_dim);
		}

		for (size_t point = 0; point < points_count; ++point)
		{
			size_t cluster = labels[point];
			for (size_t i = 0; i < points_dim; i++) {
				double xi = i;
				double yi = data[point][i];
				sum_xy[cluster] += xi * yi;
				sum_x[cluster] += xi;
				sum_y[cluster] += yi;
				sum_x_sqr[cluster] += xi * xi;
				sum_y_sqr[cluster] += yi * yi;
			}
			samples_counts[cluster] += points_dim;
		}


		for (size_t cluster = 0; cluster < number_of_clusters; cluster++) {
			double N = (double)samples_counts[cluster];
			double numerator = (N * sum_xy[cluster] - sum_x[cluster] * sum_y[cluster]);
			double denominator = (N * sum_x_sqr[cluster] - sum_x[cluster] * sum_x[cluster]);
			double coeff = numerator / denominator;
			numerator = (sum_y[cluster] * sum_x_sqr[cluster] - sum_x[cluster] * sum_xy[cluster]);
			double const_term = numerator / denominator;

			for (size_t pt = 0; pt < points_dim; pt++) {
				double y = coeff * ((double)pt) + const_term;
				means[cluster][pt] = y;
			}
		}
	}


	inline void calculate_piecewise_lined_centroids(double** data, size_t* labels, size_t points_count, size_t number_of_clusters, size_t points_dim, double** means, double** old_means, double** helper_means, size_t* counts_helper, size_t* pieces_indexes, size_t pieces_indexes_len) {
		size_t arr_size = sizeof(double) * number_of_clusters;
		size_t* samples_counts = (size_t*)malloc(sizeof(size_t) * number_of_clusters);
		double* sum_xy = (double*)malloc(arr_size);
		double* sum_x = (double*)malloc(arr_size);
		double* sum_y = (double*)malloc(arr_size);
		double* sum_x_sqr = (double*)malloc(arr_size);
		double* sum_y_sqr = (double*)malloc(arr_size);

		for (size_t cluster = 0; cluster < number_of_clusters; ++cluster)
		{
			sum_xy[cluster] = 0.0;
			sum_x[cluster] = 0.0;
			sum_y[cluster] = 0.0;
			sum_x_sqr[cluster] = 0.0;
			sum_y_sqr[cluster] = 0.0;
			samples_counts[cluster] = 0;
			memcpy(old_means[cluster], means[cluster], sizeof(double) * points_dim);
		}

		for (size_t point = 0; point < points_count; ++point)
		{
			size_t cluster = labels[point];
			for (size_t i = 0; i < points_dim; i++) {
				double xi = i;
				double yi = data[point][i];
				sum_xy[cluster] += xi * yi;
				sum_x[cluster] += xi;
				sum_y[cluster] += yi;
				sum_x_sqr[cluster] += xi * xi;
				sum_y_sqr[cluster] += yi * yi;
			}
			samples_counts[cluster] += points_dim;
		}


		for (size_t cluster = 0; cluster < number_of_clusters; cluster++) {
			double N = (double)samples_counts[cluster];
			double numerator = (N * sum_xy[cluster] - sum_x[cluster] * sum_y[cluster]);
			double denominator = (N * sum_x_sqr[cluster] - sum_x[cluster] * sum_x[cluster]);
			double coeff = numerator / denominator;
			numerator = (sum_y[cluster] * sum_x_sqr[cluster] - sum_x[cluster] * sum_xy[cluster]);
			double const_term = numerator / denominator;

			for (size_t pt = 0; pt < points_dim; pt++) {
				double y = coeff * ((double)pt) + const_term;
				means[cluster][pt] = y;
			}
		}
	}
}