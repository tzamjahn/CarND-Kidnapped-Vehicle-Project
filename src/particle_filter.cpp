/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Edited on: Sep 3, 2018
 *      By: Tyler Zamjahn
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 100;
    
    std::default_random_engine gen;
    
    std::normal_distribution<double> N_x(x,std[0]);
    std::normal_distribution<double> N_y(y,std[1]);
    std::normal_distribution<double> N_theta(theta,std[2]);
    
    for(int i = 0;i<num_particles;i++){
        Particle particle;
        particle.id = i;
        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
        particle.weight =1;
        
        particles.push_back(particle);
        weights.push_back(1);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    default_random_engine gen;
    for(int i = 0;i < num_particles; i++){
        double new_x;
        double new_y;
        double new_theta;
        if(fabs(yaw_rate) < 0.0001)
        {
            new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
            new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
            new_theta = particles[i].theta;
        }
        else
        {
            new_x =particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
            new_y =particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
            new_theta = particles[i].theta+yaw_rate*delta_t;
        }
        normal_distribution<double> N_x(new_x,std_pos[0]);
        normal_distribution<double> N_y(new_y,std_pos[1]);
        normal_distribution<double> N_theta(new_theta,std_pos[2]);
        
        particles[i].x = N_x(gen);
        particles[i].y = N_y(gen);
        particles[i].theta = N_theta(gen);
    }
    
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    double gaus_norm = (1.0/(2.0*M_PI*std_landmark[0]*std_landmark[1]));
    std::cout<<gaus_norm<<endl;
    for(int i = 0; i< num_particles;i++){
        
        std::vector<int> associations_temp;
        std::vector<double> sense_x_temp;
        std::vector<double> sense_y_temp;
        
        // Declare single_landmark:
        Map::single_landmark_s landmark_temp;
        //Create new list of landmarks within the sensor range to speed up calculations
        std::vector<Map::single_landmark_s> landmarks_in_range;
        
        for(int k = 0; k < map_landmarks.landmark_list.size();k++){
            
            double dist = sqrt(pow((particles[i].x-map_landmarks.landmark_list[k].x_f),2)+
                               pow((particles[i].y-map_landmarks.landmark_list[k].y_f),2));
            if(dist<sensor_range+5.0){
                // Set values
                landmark_temp.id_i = map_landmarks.landmark_list[k].id_i;
                landmark_temp.x_f  = map_landmarks.landmark_list[k].x_f;
                landmark_temp.y_f  = map_landmarks.landmark_list[k].y_f;
                // Add to landmark list in range:
                landmarks_in_range.push_back(landmark_temp);
            }
        }
        double weight_temp = 1.0;
        for(int j = 0; j< observations.size();j++){
            //transform sensor readings from car to map coordinates
            double x_obs_map;
            double y_obs_map;
            
            x_obs_map = particles[i].x + cos(particles[i].theta)*observations[j].x -sin(particles[i].theta)*observations[j].y;
            y_obs_map = particles[i].y + sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y;
            // Find nearest landmark to observed location
            double mu_x;
            double mu_y;
            int id1;
            double min = sensor_range+5.0;
            for(int k = 0; k < landmarks_in_range.size();k++){
                double d_temp;
                d_temp = sqrt(pow((x_obs_map-landmarks_in_range[k].x_f),2) + pow((y_obs_map -landmarks_in_range[k].y_f ),2));
                if(d_temp<min){
                    mu_x = landmarks_in_range[k].x_f;
                    mu_y = landmarks_in_range[k].y_f;
                    min = d_temp;
                    id1 = landmarks_in_range[k].id_i;
                }
            }
            if(min == sensor_range+5.0){
                weight_temp = 0;
            }
            double exponent = (pow((x_obs_map - mu_x),2))/(2.0*pow(std_landmark[0],2)) + (pow((y_obs_map - mu_y),2))/(2.0 * pow(std_landmark[1],2));
            weight_temp *= gaus_norm*exp(-exponent);
            associations_temp.push_back(id1);
            sense_x_temp.push_back(x_obs_map);
            sense_y_temp.push_back(y_obs_map);
        }
        particles[i].weight = weight_temp;
        particles[i] = SetAssociations(particles[i],associations_temp,sense_x_temp,sense_y_temp);
        weights[i]= particles[i].weight;
        
    }

}
void ParticleFilter::resample() {
    default_random_engine generator;
    discrete_distribution<int> distribution(weights.begin(),weights.end());
    
    vector<Particle> resample_particles;
    for(int i = 0;i<num_particles; i++){
        resample_particles.push_back(particles[distribution(generator)]);
        
    }
    particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
