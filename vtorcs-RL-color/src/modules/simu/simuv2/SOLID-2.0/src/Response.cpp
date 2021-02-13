/*
  SOLID - Software Library for Interference Detection
  Copyright (C) 1997-1998  Gino van den Bergen

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Please send remarks, questions and bug reports to gino@win.tue.nl,
  or write to:
                  Gino van den Bergen
		  Department of Mathematics and Computing Science
		  Eindhoven University of Technology
		  P.O. Box 513, 5600 MB Eindhoven, The Netherlands
*/

#ifdef _MSC_VER
#pragma warning(disable:4786) // identifier was truncated to '255'
#endif // _MSC_VER

#include "Response.h"

DtCollData Response::coll_data;

void Response::operator()(DtObjectRef a, DtObjectRef b, 
			  const Point& pa, const Point& pb, 
			  const Vector& v) const {  
  coll_data.point1[X] = pa[X];
  coll_data.point1[Y] = pa[Y];
  coll_data.point1[Z] = pa[Z];
  coll_data.point2[X] = pb[X];
  coll_data.point2[Y] = pb[Y];
  coll_data.point2[Z] = pb[Z];
  coll_data.normal[X] = v[X];
  coll_data.normal[Y] = v[Y];
  coll_data.normal[Z] = v[Z];
  response(client_data, a, b, &coll_data); 
}

void Response::operator()(DtObjectRef a, DtObjectRef b) const {  
  response(client_data, a, b, 0); 
}
