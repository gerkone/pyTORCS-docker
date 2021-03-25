/***************************************************************************

    file                 : pitmenu.cpp
    created              : Mon Apr 24 18:16:37 CEST 2000
    copyright            : (C) 2013 by Eric Espie, Bernhard Wymann
    email                : torcs@free.fr
    version              : $Id: pitmenu.cpp,v 1.4.2.8 2014/05/20 11:02:18 berniw Exp $

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/** @file   
    Pit menu
    @author Bernhard Wymann, Eric Espie
    @version $Id: pitmenu.cpp,v 1.4.2.8 2014/05/20 11:02:18 berniw Exp $
*/

#include <stdlib.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <tgfclient.h>
#include <car.h>
#include <portability.h>
#include <racescreens.h>

static void *menuHandle = NULL;
static int fuelId;
static int repairId;
static tCarElt *rmCar;
static tRmInfo *rmInfo;


static void rmUpdtFuel(void * /* dummy */)
{
	char *val;
	const int BUFSIZE = 32;
	char buf[BUFSIZE];

	val = GfuiEditboxGetString(menuHandle, fuelId);
	rmCar->pitcmd.fuel = (tdble)strtol(val, (char **)NULL, 0);
	snprintf(buf, BUFSIZE, "%f", rmCar->pitcmd.fuel);
	GfuiEditboxSetString(menuHandle, fuelId, buf);
}


static void rmUpdtRepair(void * /* dummy */)
{
	char *val;
	const int BUFSIZE = 32;
	char buf[BUFSIZE];

	val = GfuiEditboxGetString(menuHandle, repairId);
	rmCar->pitcmd.repair = strtol(val, (char **)NULL, 0);
	snprintf(buf, BUFSIZE, "%d", rmCar->pitcmd.repair);
	GfuiEditboxSetString(menuHandle, repairId, buf);
}


static tfuiCallback rmCallback;
static void *rmUserData;


static void rmStopAndGo(void * /* dummy */)
{
	rmCar->_pitStopType = RM_PIT_STOPANDGO;
	rmCallback(rmUserData);
}


static void rmRepair(void* /* dummy */)
{
	rmCar->_pitStopType = RM_PIT_REPAIR;
	rmCallback(rmUserData);
}


/** @brief Pit menu
 *  @ingroup racemantools
 *  @param[in,out] car Car data is returned in tCarElt.pitcmd
 *  @param[in] reInfo Required for passing data to car setup screen, RmCarSetupScreenInit
 *  @param[in] userdata Userdata for callback
 *  @param[in] callback Callback to take action on the changes made
 */ 
void RmPitMenuStart(tCarElt *car, tRmInfo *reInfo, void *userdata, tfuiCallback callback)
{
	const int BUFSIZE = 256;
	char buf[BUFSIZE];
	int	y, x, dy;

	rmCar = car;
	rmInfo = reInfo;

	if (menuHandle) {
		GfuiScreenRelease(menuHandle);
	}
	menuHandle = GfuiMenuScreenCreate("Pit Stop Info");

	x = 80;
	y = 380;
	snprintf(buf, BUFSIZE, "Driver: %s", car->_name);
	GfuiLabelCreate(menuHandle, buf, GFUI_FONT_LARGE_C, x, y, GFUI_ALIGN_HL_VB, 0);
	dy = GfuiFontHeight(GFUI_FONT_LARGE_C) + 5;

	y -= dy;
	snprintf(buf, BUFSIZE, "Remaining Laps: %d", car->_remainingLaps);
	GfuiLabelCreate(menuHandle, buf, GFUI_FONT_MEDIUM_C, x, y, GFUI_ALIGN_HL_VB, 0);

	y -= dy;
	snprintf(buf, BUFSIZE, "Remaining Fuel: %.1f l", car->_fuel);
	GfuiLabelCreate(menuHandle, buf, GFUI_FONT_MEDIUM_C, x, y, GFUI_ALIGN_HL_VB, 0);

	y -= dy;
	GfuiLabelCreate(menuHandle, "Fuel amount (liters):", GFUI_FONT_MEDIUM_C, x, y, GFUI_ALIGN_HL_VB, 0);

	int dx = GfuiFontWidth(GFUI_FONT_MEDIUM_C, "Fuel amount (liters)") + 20;
	snprintf(buf, BUFSIZE, "%d", (int)car->pitcmd.fuel);
	fuelId = GfuiEditboxCreate(menuHandle, buf, GFUI_FONT_MEDIUM_C,
					x + dx, y,
					0, 10, NULL, (tfuiCallback)NULL, rmUpdtFuel);

	y -= dy;
	GfuiLabelCreate(menuHandle, "Repair amount:", GFUI_FONT_MEDIUM_C, x, y, GFUI_ALIGN_HL_VB, 0);

	snprintf(buf, BUFSIZE, "%d", (int)car->pitcmd.repair);
	repairId = GfuiEditboxCreate(menuHandle, buf, GFUI_FONT_MEDIUM_C,
					x + dx, y,
					0, 10, NULL, (tfuiCallback)NULL, rmUpdtRepair);


	GfuiButtonCreate(menuHandle, "Repair", GFUI_FONT_LARGE, 160, 40, 130, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				NULL, rmRepair, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
	rmCallback = callback;
	rmUserData = userdata;

	int buttonid;
	// Just enable stop and go if penalty is pending
	buttonid = GfuiButtonCreate(menuHandle, "Stop & Go", GFUI_FONT_LARGE, 320, 40, 130, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				NULL, rmStopAndGo, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
	tCarPenalty *penalty = GF_TAILQ_FIRST(&(car->_penaltyList));
	if (!penalty || (penalty && penalty->penalty != RM_PENALTY_STOPANDGO)) {
		GfuiEnable(menuHandle, buttonid, GFUI_DISABLE);
	}

	// Just enable car setup button in practice and qualifying sessions
	buttonid = GfuiButtonCreate(menuHandle, "Setup", GFUI_FONT_LARGE, 480, 40, 130, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
		RmCarSetupScreenInit(menuHandle, rmCar, rmInfo), GfuiScreenActivate, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
	if (reInfo->s->raceInfo.type != RM_TYPE_PRACTICE && reInfo->s->raceInfo.type != RM_TYPE_QUALIF) {
		GfuiEnable(menuHandle, buttonid, GFUI_DISABLE);
	}
	
	GfuiScreenActivate(menuHandle);
}
