/*
 * label.h
 *
 * Labels for classes (int + string)
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

#ifndef xmm_lib_label__
#define xmm_lib_label__

#include <iostream>
#include "xmm_common.h"
#include "json_utilities.h"

namespace xmm
{
    /**
     * @ingroup TrainingSet
     * @class Label
     * @brief Label of a data phrase
     * @details Possible types are int and string
     */
    class Label : public Writable
    {
    public:
        /**
         * @brief Type of the Label: 'INT' or 'SYM'
         */
        enum {INT, SYM} type;
        
        /**
         * @brief Default constructor
         * @details The default label type is INT, with value 0
         */
        Label();
        
        /**
         * @brief Destructor
         */
        virtual ~Label() {}
        
        /**
         * @brief Constructor.
         * @details The default label type is INT, with value 0
         * @param l integer label
         */
        explicit Label(int l);
        
        /**
         * @brief Constructor from C++ std::string
         * @param l symbolic label
         */
        explicit Label(std::string l);
        
        /**
         * @brief Constructor from C-like string
         * @param l symbolic label
         */
        explicit Label(char* l);
        
        /**
         * @brief Copy Constructor
         */
        Label(Label const& src);
        
        /**
         * @brief Assignment
         * @param src source Label
         */
        Label& operator=(Label const& src);
        
        /**
         * @brief Assignment from int
         * @param l integer label
         */
        Label& operator=(int l);
        
        /**
         * @brief Assignment from std::string
         * @param l symbolic label
         */
        Label& operator=(std::string l);
        
        /**
         * @brief Assignment from C-like string
         * @param l symbolic label as C-string
         */
        Label& operator=(char* l);
        
        /**
         * @brief Check label equality
         * @param src source Label
         * @return true if labels are equal (type and value)
         */
        bool operator==(Label const& src) const;
        
        /**
         * @brief Check label inequality
         * @param src source Label
         * @return true if labels are different (type or value)
         */
        bool operator!=(Label const& src) const;
        
        /**
         * @brief Check label inequality
         * @param src source Label
         * @return true if the label is inferior to source (alphabetical order is used for symbolic labels)
         */
        bool operator<(Label const& src) const;
        
        /**
         * @brief Check label inequality
         * @param src source Label
         * @return true if the label is inferior or equal to source (alphabetical order is used for symbolic labels)
         */
        bool operator<=(Label const& src) const;
        
        /**
         * @brief Check label inequality
         * @param src source Label
         * @return true if the label is superior to source (alphabetical order is used for symbolic labels)
         */
        bool operator>(Label const& src) const;
        
        /**
         * @brief Check label inequality
         * @param src source Label
         * @return true if the label is superior or equal to source (alphabetical order is used for symbolic labels)
         */
        bool operator>=(Label const& src) const;
        
        /**
         * @brief Get integer label value
         * @throw runtime_error if label type is not INT
         * @return integer label
         */
        int getInt() const;
        
        /**
         * @brief Get symbolic label value
         * @throw runtime_error if label type is not SYM
         * @return symbolic label
         */
        std::string getSym() const;
        
        /**
         * @brief Set integer label value => sets label type to INT
         * @param l integer label
         */
        void setInt(int l);
        
        /**
         * @brief Try to set an integer from a string that contains one.
         * @param l integer label stored in a string
         * @return true if the integer label could be set
         */
        bool trySetInt(std::string l);
        
        /**
         * @brief Set symbolic label value => sets label type to SYM
         * @param l symbolic label
         */
        void setSym(std::string l);
        
        /**
         * @brief Set symbolic label value => sets label type to SYM
         * @param l symbolic label as C-string
         */
        void setSym(char* l);
        
        /**
         * @brief Write to JSON Node
         * @return JSON Node containing the label information
         */
        JSONNode to_json() const;
        
        /**
         * @brief Read from JSON Node
         * @param root JSON Node containing the label information
         */
        void from_json(JSONNode root);
        
        /**
         * @brief print label as c++ std::string
         */
        std::string as_string() const;
        
        /**
         * @brief Insertion operator
         * @param stream output stream
         * @param l label
         */
        friend std::ostream& operator<<(std::ostream& stream, xmm::Label const& l);
        
    protected:
        ///@cond DEVDOC
        /**
         * @brief Integer value
         * @details [long description]
         */
        int intLabel_;
        
        /**
         * @brief symbolic value
         */
        std::string symLabel_;
        
        ///@endcond
    };
    
    ///@cond DEVDOC
    /**
     * @brief Check if the string contains an integer
     * @param s std::string to check
     * @return true if the string contains an integer
     */
    bool is_number(const std::string& s);
    
    /**
     * @brief Get integer from string
     */
    int to_int(const std::string& s);
    
    /**
     * @brief Insertion operator
     * @param stream output stream
     * @param l label
     */
    std::ostream& operator<<(std::ostream& stream, xmm::Label const& l);
    
    ///@endcond
}

#endif
