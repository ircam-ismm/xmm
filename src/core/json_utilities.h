/*
 * json_utilities.h
 *
 * Set of utility functions for JSON I/O
 *
 * Contact:
 * - Jules Françoise <jules.francoise@ircam.fr>
 *
 * This code has been initially authored by Jules Françoise
 * <http://julesfrancoise.com> during his PhD thesis, supervised by Frédéric
 * Bevilacqua <href="http://frederic-bevilacqua.net>, in the Sound Music
 * Movement Interaction team <http://ismm.ircam.fr> of the
 * STMS Lab - IRCAM, CNRS, UPMC (2011-2015).
 *
 * Copyright (C) 2015 UPMC, Ircam-Centre Pompidou.
 *
 * This File is part of XMM.
 *
 * XMM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * XMM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with XMM.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef xmm_lib_json_utilities_h
#define xmm_lib_json_utilities_h

#include <iostream>
#include <sstream>
#include <string>
#include <exception>
#include <vector>
#include "libjson.h"

namespace xmm
{
    /**
     * @ingroup Utilities
     * @class JSONException
     * @brief Simple Exception class for handling JSON I/O errors
     */
    class JSONException : public std::exception
    {
    public:
        /**
         * @brief Default Constructor
         * @param message error message
         * @param nodename name of the JSON node where the error occurred
         */
        JSONException(std::string message, std::string nodename="")
        : message_(message)
        {
            nodename_.push_back(nodename);
        }
        
        /**
         * @brief Constructor From exception message
         * @param src Source Exception
         * @param nodename name of the
         */
        explicit JSONException(exception const& src, std::string nodename)
        : message_(src.what())
        {
            nodename_.push_back(nodename);
        }
        
        /**
         * @brief Constructor From exception message
         * @param src Source Exception
         * @param nodename name of the
         */
        explicit JSONException(JSONException const& src, std::string nodename)
        {
            this->_copy(this, src);
            nodename_.push_back(nodename);
        }
        
        /**
         * @brief Copy Constructor
         * @param src Source JSON exception
         */
        JSONException(JSONException const& src)
        {
            this->_copy(this, src);
        }
        
        /**
         * @brief Assigment
         * @param src Source JSON exception
         */
        JSONException& operator=(JSONException const& src)
        {
            if(this != &src)
            {
                _copy(this, src);
            }
            return *this;
        }
        
        /**
         * @brief Copy between two JSON exceptions
         * @param dst destination JSON exception
         * @param src Source JSON exception
         */
        virtual void _copy(JSONException *dst,
                           JSONException const& src)
        
        {
            dst->message_ = src.message_;
            dst->nodename_ = src.nodename_;
        }
        
        /**
         * @brief Destructor
         */
        virtual ~JSONException() throw()
        {}
        
        /**
         * @brief Get exception message
         * @return exception message
         */
        virtual const char * what() const throw()
        {
            std::stringstream fullmsg;
            fullmsg << "Error reading JSON, Message: " + message_ + " // (Node List: ";
            for (unsigned int i = static_cast<unsigned int>(nodename_.size()) - 1 ; i > 0 ; --i)
                fullmsg << nodename_[i] << " > ";
            fullmsg << nodename_[0];
            fullmsg << ")";
            return strdup(fullmsg.str().c_str());
        }
        
    private:
        std::string message_;
        std::vector<std::string> nodename_;
    };
    
    ///@{
    /**
     * @name JSON conversion to/from arrays and vectors
     */
    template <typename T>
    JSONNode array2json(T const* a, int n, std::string name="array")
    {
        JSONNode json_data(JSON_ARRAY);
        json_data.set_name(name);
        for (int i=0 ; i<n ; i++) {
            json_data.push_back(JSONNode("", a[i]));
        }
        return json_data;
    }
    
    template <typename T>
    void json2array(JSONNode root, T* a, int n)
    {
        if (root.type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
        unsigned int i = 0;
        for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
        {
            if (i >= n)
                throw JSONException("JSON 2 Array: Index out of bounds");
            if (array_it->type() != JSON_NUMBER)
                throw JSONException("JSON 2 Vector: Wrong type");
            a[i++] = array_it->as_int();
        }
    }
    
    template <> void json2array(JSONNode root, float* a, int n);
    template <> void json2array(JSONNode root, double* a, int n);
    template <> void json2array(JSONNode root, bool* a, int n);
    template <> void json2array(JSONNode root, std::string* a, int n);
    
    template <typename T>
    JSONNode vector2json(std::vector<T> const& a, std::string name="array")
    {
        JSONNode json_data(JSON_ARRAY);
        json_data.set_name(name);
        for (size_t i=0 ; i<a.size() ; i++) {
            json_data.push_back(JSONNode("", a[i]));
        }
        return json_data;
    }
    
    template <typename T>
    void json2vector(JSONNode root, std::vector<T>& a, int n)
    {
        if (root.type() != JSON_ARRAY)
            throw JSONException("Wrong type: was expecting 'JSON_ARRAY'", root.name());
        unsigned int i = 0;
        for (JSONNode::const_iterator array_it = root.begin(); array_it != root.end(); ++array_it)
        {
            if (i >= n)
                throw JSONException("JSON 2 Vector: Index out of bounds");
            if (array_it->type() != JSON_NUMBER)
                throw JSONException("JSON 2 Vector: Wrong type");
            a[i++] = array_it->as_int();
        }
    }
    
    template <> void json2vector(JSONNode root, std::vector<float>& a, int n);
    template <> void json2vector(JSONNode root, std::vector<double>& a, int n);
    template <> void json2vector(JSONNode root, std::vector<bool>& a, int n);
    template <> void json2vector(JSONNode root, std::vector<std::string>& a, int n);
    
    ///@}
}

#endif
