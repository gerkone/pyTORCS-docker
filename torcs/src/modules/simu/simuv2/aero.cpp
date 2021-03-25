/***************************************************************************

    file                 : aero.cpp
    created              : Sun Mar 19 00:04:50 CET 2000
    copyright            : (C) 2000-2013 by Eric Espie, Bernhard Wymann
    email                : torcs@free.fr
    version              : $Id: aero.cpp,v 1.14.2.5 2014/02/10 10:06:30 berniw Exp $

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/



#include "sim.h"

void SimAeroConfig(tCar *car)
{
	void *hdle = car->params;
	tdble Cx, FrntArea;

	Cx       = GfParmGetNum(hdle, SECT_AERODYNAMICS, PRM_CX, (char*)NULL, 0.4f);
	FrntArea = GfParmGetNum(hdle, SECT_AERODYNAMICS, PRM_FRNTAREA, (char*)NULL, 2.5f);
	car->aero.Clift[0] = GfParmGetNum(hdle, SECT_AERODYNAMICS, PRM_FCL, (char*)NULL, 0.0f);
	car->aero.Clift[1] = GfParmGetNum(hdle, SECT_AERODYNAMICS, PRM_RCL, (char*)NULL, 0.0f);
	car->aero.SCx2 = 0.645f * Cx * FrntArea;
	car->aero.Cd += car->aero.SCx2;
}


void  SimAeroUpdate(tCar *car, tSituation *s)
{
	tdble hm;
	int i;    
	tCar *otherCar;
	tdble x, y;
	tdble yaw, otherYaw, airSpeed, tmpas, spdang, tmpsdpang, dyaw;
	tdble dragK = 1.0;

	x = car->DynGCg.pos.x;
	y = car->DynGCg.pos.y;
	yaw = car->DynGCg.pos.az;
	airSpeed = car->DynGC.vel.x;
	spdang = atan2(car->DynGCg.vel.y, car->DynGCg.vel.x);

    if (airSpeed > 10.0f) {
		for (i = 0; i < s->_ncars; i++) {
			if (i == car->carElt->index) {
				// skip myself
				continue;
			}
			
			otherCar = &(SimCarTable[i]);
			otherYaw = otherCar->DynGCg.pos.az;
			tmpsdpang = spdang - atan2(y - otherCar->DynGCg.pos.y, x - otherCar->DynGCg.pos.x);
			NORM_PI_PI(tmpsdpang);
			dyaw = yaw - otherYaw;
			NORM_PI_PI(dyaw);
			
			if ((otherCar->DynGC.vel.x > 10.0f) && (fabs(dyaw) < 0.1396f)) {
				if (fabs(tmpsdpang) > 2.9671f) {	    /* 10 degrees */
					// behind another car
					tmpas = 1.0f - exp(- 2.0f * DIST(x, y, otherCar->DynGCg.pos.x, otherCar->DynGCg.pos.y) /
									  (otherCar->aero.Cd * otherCar->DynGC.vel.x));
					if (tmpas < dragK) {
						dragK = tmpas;
					}
				} else if (fabs(tmpsdpang) < 0.1396f) {	    /* 8 degrees */
					// before another car, lower drag by maximum 15% (this is just another guess)
					tmpas = 1.0f - 0.15f * exp(- 8.0f * DIST(x, y, otherCar->DynGCg.pos.x, otherCar->DynGCg.pos.y) / (car->aero.Cd * car->DynGC.vel.x));
					if (tmpas < dragK) {
						dragK = tmpas;
					}
				}
			}
		}
    }

	car->airSpeed2 = airSpeed * airSpeed;
	tdble v2 = car->airSpeed2;
	
	// simulate ground effect drop off caused by non-frontal airflow (diffusor stops working etc.)
	tdble cosa = 1.0f;	
	if (car->speed > 1.0f) {
		cosa = car->DynGC.vel.x/car->speed;
	}
	
	if (cosa < 0.0f) {
		cosa = 0.0f;
	}
			
	car->aero.drag = -SIGN(car->DynGC.vel.x) * car->aero.SCx2 * v2 * (1.0f + (tdble)car->dammage / 10000.0f) * dragK * dragK;

	hm = 1.5f * (car->wheel[0].rideHeight + car->wheel[1].rideHeight + car->wheel[2].rideHeight + car->wheel[3].rideHeight);
	hm = hm*hm;
	hm = hm*hm;
	hm = 2.0f * exp(-3.0f*hm);
	car->aero.lift[0] = - car->aero.Clift[0] * v2 * hm * cosa;
	car->aero.lift[1] = - car->aero.Clift[1] * v2 * hm * cosa;
}

static const char *WingSect[2] = {SECT_FRNTWING, SECT_REARWING};

void SimWingConfig(tCar *car, int index)
{
	void *hdle = car->params;
	tWing *wing = &(car->wing[index]);
	tdble area;

	area              = GfParmGetNum(hdle, WingSect[index], PRM_WINGAREA, (char*)NULL, 0);
	wing->angle       = GfParmGetNum(hdle, WingSect[index], PRM_WINGANGLE, (char*)NULL, 0);
	wing->staticPos.x = GfParmGetNum(hdle, WingSect[index], PRM_XPOS, (char*)NULL, 0);
	wing->staticPos.z = GfParmGetNum(hdle, WingSect[index], PRM_ZPOS, (char*)NULL, 0);
	wing->staticPos.x -= car->statGC.x;
	
	wing->Kx = -1.23f * area;
	wing->Kz = 4.0f * wing->Kx;

	if (index == 1) {
		car->aero.Cd -= wing->Kx*sin(wing->angle);
	}
}


void SimWingReConfig(tCar *car, int index)
{
	tCarPitSetupValue* v = &car->carElt->pitcmd.setup.wingangle[index];
	if (SimAdjustPitCarSetupParam(v)) {
		tWing *wing = &(car->wing[index]);
		tdble oldCd = wing->Kx*sin(wing->angle);
		wing->angle = v->value;
		
		if (index == 1) {
			car->aero.Cd += oldCd;
			car->aero.Cd -= wing->Kx*sin(wing->angle);
		}
	}
}


void SimWingUpdate(tCar *car, int index, tSituation* s)
{
	tWing  *wing = &(car->wing[index]);
	tdble vt2 = car->airSpeed2;
	// compute angle of attack
	tdble aoa = atan2(car->DynGC.vel.z, car->DynGC.vel.x) + car->DynGCg.pos.ay;
	aoa += wing->angle;
	// the sinus of the angle of attack
	tdble sinaoa = sin(aoa);

	if (car->DynGC.vel.x > 0.0f) {
		wing->forces.x = wing->Kx * vt2 * (1.0f + (tdble)car->dammage / 10000.0f) * sinaoa;
		wing->forces.z = wing->Kz * vt2 * sinaoa;
	} else {
		wing->forces.x = wing->forces.z = 0.0f;
	}
}