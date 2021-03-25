/***************************************************************************

    file        : raceresults.cpp
    created     : Thu Jan  2 12:43:10 CET 2003
    copyright   : (C) 2002, 2014 by Eric Espie, Bernhard Wymann                        
    email       : eric.espie@torcs.org   
    version     : $Id: raceresults.cpp,v 1.9.2.15 2014/05/23 08:38:31 berniw Exp $                                  

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
    Processing of race results		
    @author	Bernhard Wymann, Eric Espie
    @version $Id: raceresults.cpp,v 1.9.2.15 2014/05/23 08:38:31 berniw Exp $
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <tgfclient.h>
#include <robot.h>
#include <raceman.h>
#include <racescreens.h>
#include <robottools.h>
#include <portability.h>

#include "racemain.h"
#include "racegl.h"
#include "raceinit.h"
#include "raceengine.h"

#include "raceresults.h"

typedef struct
{
	char *carName;
	char	*modName;
	int		drvIdx;
	int		points;
} tReStandings;


/*
	Applies pending penalties after the race:
	- Time penalties
	- Pending stop & go (can happen if penalty is given within the last 5 laps)
	- Pending drive through (dito)

	I assume that the average track speed is 300km/h (v0), and that the pit entry
	and exit costs 10s. We call the pit speed limit v1, and the length of the pit
	lane s. So the time lost with driving through the pit lane is
	dt = s*(v0-v1)/(v0*v1). We assume that stopping costs additional 6 seconds.  
	
	Drive through: 10 + dt [s]
	Stop & Go: 10 + 6 + dt [s]

	Applying time penalties with cars in different laps is problematic, picture
	this situation: We have 3 cars running, pos 1 car is right behind the pos 3
	car (so it will overlap in a moment), pos 3 car is right behind pos 2 car,
	and pos 2 car has a 30s penalty pending. Now right before the finish line the
	pos 1 car passes (overlaps) the pos 3 car, so the race is done for pos 1 car,
	and a moment later for the pos 3 car. But the pos 2 car can finish its last
	lap, although it has a 30s penalty pending, which would have thrown it behind
	the pos 3 car, if the overlapping not had happened.

	So just adding times is not enough (btw. the FIA does just add times... weird).
	
	All time penalties are added up and then applied to the ranking like this:
	We "pump up" the result of the opponents with less laps up to our laps
	with a simple linear model: opponent_race_time/opponent_laps*our_race_laps, so
	we are worse when:
	our_race_time + our_penalty_time > 
	opponent_race_time/opponent_laps*our_race_laps + opponent_penalty_time
	
	This works as well for opponents in the same lap, opponent_laps*our_race_laps is
	then 1. The calculation requires that at least one full lap has been driven by
	the drivers which are compared.

	Wrecked cars are considered worse than a penalty.
*/
static void ReApplyRaceTimePenalties(void)
{
	// First update all penalty times, apply pending drive through/stop and go
	int i;
	tSituation *s = ReInfo->s;
	tCarPenalty *penalty;
	tCarElt* car;

	if (ReInfo->track->pits.type == TR_PIT_ON_TRACK_SIDE) {
		const tdble drivethrough = 10.0f;
		const tdble stopandgo = 6.0f + drivethrough;

		const tdble v0 = 84.0f;
		tdble v1 = ReInfo->track->pits.speedLimit;
		tdble dv = v0 - v1;
		tdble dt = 0.0;

		if (dv > 1.0f && v1 > 1.0f) {
			dt = (ReInfo->track->pits.nMaxPits*ReInfo->track->pits.len)*dv/(v0*v1);
		}

		for (i = 0; i < s->_ncars; i++) {
			car = s->cars[i];
			penalty = GF_TAILQ_FIRST(&(car->_penaltyList));
			while (penalty) {
				if (penalty->penalty == RM_PENALTY_DRIVETHROUGH) {
					car->_penaltyTime += dt + drivethrough;
				} else if (penalty->penalty == RM_PENALTY_STOPANDGO) {
					car->_penaltyTime += dt + stopandgo;
				} else {
					GfError("Unknown penalty.");
				}
				penalty = GF_TAILQ_NEXT(penalty, link);
			}		
		}
	}

	// Now sort the cars taking into account the penalties
	int j;
	for (i = 1; i < s->_ncars; i++) {
		j = i;
		while (j > 0) {
			// Order without penalties is already ok, so if there is no penalty we do not move down
			if (s->cars[j-1]->_penaltyTime > 0.0f) {
				int l1 = MIN(s->cars[j-1]->_laps, s->_totLaps + 1) - 1;
				int l2 = MIN(s->cars[j]->_laps, s->_totLaps + 1) - 1;
				// If the drivers did not at least complete one lap we cannot apply the rule, check.
				// If the cars are wrecked we do not care about penalties.
				if (
					l1 < 1 ||
					l2 < 1 ||
					(s->raceInfo.maxDammage < s->cars[j-1]->_dammage) ||
					(s->raceInfo.maxDammage < s->cars[j]->_dammage))
				{
					// Because the cars came already presorted, all following cars must be even worse,
					// so we can break the iteration here.
					i = s->_ncars;	// Break outer loop
					break;			// Break inner loop
				}
				
				tdble t1 = s->cars[j-1]->_curTime + s->cars[j-1]->_penaltyTime;
				tdble t2 = s->cars[j]->_curTime*tdble(l1)/tdble(l2) + s->cars[j]->_penaltyTime;

				if (t1 > t2) {
					// Swap
					car = s->cars[j];
					s->cars[j] = s->cars[j-1];
					s->cars[j-1] = car;
					s->cars[j]->_pos = j+1;
					s->cars[j-1]->_pos = j;
					j--;
					continue;
				}
			}
			j = 0;
		}
	}
}


void
ReInitResults(void)
{
	struct tm *stm;
	time_t t;
	void *results;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE];
	
	t = time(NULL);
	stm = localtime(&t);
	snprintf(buf, BUFSIZE, "%sresults/%s/results-%4d-%02d-%02d-%02d-%02d-%02d.xml",
		GetLocalDir(),
		ReInfo->_reFilename,
		stm->tm_year+1900,
		stm->tm_mon+1,
		stm->tm_mday,
		stm->tm_hour,
		stm->tm_min,
		stm->tm_sec
	);
	
	ReInfo->results = GfParmReadFile(buf, GFPARM_RMODE_STD | GFPARM_RMODE_CREAT);
	results = ReInfo->results;
	GfParmSetNum(results, RE_SECT_HEADER, RE_ATTR_DATE, NULL, (tdble)t);
	GfParmSetNum(results, RE_SECT_CURRENT, RE_ATTR_CUR_TRACK, NULL, 1);
	GfParmSetNum(results, RE_SECT_CURRENT, RE_ATTR_CUR_RACE, NULL, 1);
	GfParmSetNum(results, RE_SECT_CURRENT, RE_ATTR_CUR_DRIVER, NULL, 1);	
}

void
ReEventInitResults(void)
{
	int nCars;
	int i;
	void *results = ReInfo->results;
	void *params = ReInfo->params;
	const int BUFSIZE = 1024;
	char path[BUFSIZE], path2[BUFSIZE];
	
	nCars = GfParmGetEltNb(params, RM_SECT_DRIVERS);
	for (i = 1; i < nCars + 1; i++) {
		snprintf(path, BUFSIZE, "%s/%s/%d", ReInfo->track->name, RM_SECT_DRIVERS, i);
		snprintf(path2, BUFSIZE, "%s/%d", RM_SECT_DRIVERS, i);
		GfParmSetStr(results, path, RE_ATTR_DLL_NAME, GfParmGetStr(params, path2, RM_ATTR_MODULE, ""));
		GfParmSetNum(results, path, RE_ATTR_INDEX, NULL, GfParmGetNum(params, path2, RM_ATTR_IDX, (char*)NULL, 0));
    }
}

void
ReUpdateStandings(void)
{
	int maxDrv;
	int curDrv;
	int runDrv;
	char *modName;
	int drvIdx;
	int points;
	int i, j;
	int found;
	tReStandings	*standings = 0;
	void *results = ReInfo->results;
	const int BUFSIZE = 1024;
	char str1[BUFSIZE], str2[BUFSIZE], path[BUFSIZE], path2[BUFSIZE];
	
	snprintf(path, BUFSIZE, "%s/%s/%s/%s", ReInfo->track->name, RE_SECT_RESULTS, ReInfo->_reRaceName, RE_SECT_RANK);
	
	runDrv = GfParmGetEltNb(results, path);
	curDrv = GfParmGetEltNb(results, RE_SECT_STANDINGS);
	maxDrv = curDrv + runDrv;
	
	standings = (tReStandings *)calloc(maxDrv, sizeof(tReStandings));
	
	/* Read the current standings */
	for (i = 0; i < curDrv; i++) {
		snprintf(path2, BUFSIZE, "%s/%d", RE_SECT_STANDINGS, i + 1);
		standings[i].carName = strdup(GfParmGetStr(results, path2, RE_ATTR_NAME, 0));
		standings[i].modName = strdup(GfParmGetStr(results, path2, RE_ATTR_MODULE, 0));
		standings[i].drvIdx  = (int)GfParmGetNum(results, path2, RE_ATTR_IDX, NULL, 0);
		standings[i].points  = (int)GfParmGetNum(results, path2, RE_ATTR_POINTS, NULL, 0);
	}

	GfParmListClean(results, RE_SECT_STANDINGS);
	
	for (i = 0; i < runDrv; i++) {
		/* Search the driver in the standings */
		found = 0;
		snprintf(path, BUFSIZE, "%s/%s/%s/%s/%d", ReInfo->track->name, RE_SECT_RESULTS, ReInfo->_reRaceName, RE_SECT_RANK, i + 1);
		const char* carName = GfParmGetStr(results, path, RE_ATTR_NAME, 0);
		for (j = 0; j < curDrv; j++) {
			if (!strcmp(carName, standings[j].carName)) {
				found = 1;
				break;
			}
		}

		if (!found) {
			/* Add the new driver */
			curDrv++;
			standings[j].carName = strdup(carName);
			standings[j].modName = strdup(GfParmGetStr(results, path, RE_ATTR_MODULE, 0));
			standings[j].drvIdx  = (int)GfParmGetNum(results, path, RE_ATTR_IDX, NULL, 0);
			standings[j].points  = (int)GfParmGetNum(results, path, RE_ATTR_POINTS, NULL, 0);
		} else {
			/* Add the new points */
			standings[j].points += (int)GfParmGetNum(results, path, RE_ATTR_POINTS, NULL, 0);
		}
		/* bubble sort... */
		while (j > 0) {
			if (standings[j - 1].points >= standings[j].points) {
				break;
			}
			/* Swap with preceeding */
			char* tmpCarName;
			tmpCarName = standings[j].carName;
			modName = standings[j].modName;
			drvIdx  = standings[j].drvIdx;
			points  = standings[j].points;
		
			standings[j].carName = standings[j - 1].carName;
			standings[j].modName = standings[j - 1].modName;
			standings[j].drvIdx  = standings[j - 1].drvIdx;
			standings[j].points  = standings[j - 1].points;
		
			standings[j - 1].carName = tmpCarName;
			standings[j - 1].modName = modName;
			standings[j - 1].drvIdx  = drvIdx;
			standings[j - 1].points  = points;
		
			j--;
		}
	}
	
	/* Store the standing back */
	for (i = 0; i < curDrv; i++) {
		snprintf(path, BUFSIZE, "%s/%d", RE_SECT_STANDINGS, i + 1);
		GfParmSetStr(results, path, RE_ATTR_NAME, standings[i].carName);
		free(standings[i].carName);
		GfParmSetStr(results, path, RE_ATTR_MODULE, standings[i].modName);
		free(standings[i].modName);
		GfParmSetNum(results, path, RE_ATTR_IDX, NULL, standings[i].drvIdx);
		GfParmSetNum(results, path, RE_ATTR_POINTS, NULL, standings[i].points);
	}
	free(standings);
	
	snprintf(str1, BUFSIZE, "%sconfig/params.dtd", GetDataDir());
	snprintf(str2, BUFSIZE, "<?xml-stylesheet type=\"text/xsl\" href=\"file:///%sconfig/style.xsl\"?>", GetDataDir());
	
	GfParmSetDTD (results, str1, str2);
	GfParmCreateDirectory(0, results);
	GfParmWriteFile(0, results, "Results");
}


void ReStoreRaceResults(const char *race)
{
	int i;
	int nCars;
	tCarElt *car;
	tSituation *s = ReInfo->s;
	char *carName;
	void *carparam;
	void *results = ReInfo->results;
	void *params = ReInfo->params;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE], path[BUFSIZE], path2[BUFSIZE];
	
	/* Store the number of laps of the race */
	switch (ReInfo->s->_raceType) {
		case RM_TYPE_RACE:
			car = s->cars[0];
			if (car->_laps > s->_totLaps) car->_laps = s->_totLaps + 1;

			snprintf(path, BUFSIZE, "%s/%s/%s", ReInfo->track->name, RE_SECT_RESULTS, race);
			GfParmListClean(results, path);
			GfParmSetNum(results, path, RE_ATTR_LAPS, NULL, car->_laps - 1);
			
			ReApplyRaceTimePenalties();

			for (i = 0; i < s->_ncars; i++) {
				snprintf(path, BUFSIZE, "%s/%s/%s/%s/%d", ReInfo->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK, i + 1);
				car = s->cars[i];
				if (car->_laps > s->_totLaps) car->_laps = s->_totLaps + 1;
			
				GfParmSetStr(results, path, RE_ATTR_NAME, car->_name);
			
				snprintf(buf, BUFSIZE, "cars/%s/%s.xml", car->_carName, car->_carName);
				carparam = GfParmReadFile(buf, GFPARM_RMODE_STD);
				carName = GfParmGetName(carparam);
			
				GfParmSetStr(results, path, RE_ATTR_CAR, carName);
				GfParmSetNum(results, path, RE_ATTR_INDEX, NULL, car->index);
			
				GfParmSetNum(results, path, RE_ATTR_LAPS, NULL, car->_laps - 1);
				GfParmSetNum(results, path, RE_ATTR_TIME, NULL, car->_curTime + car->_penaltyTime);
				GfParmSetNum(results, path, RE_ATTR_PENALTYTIME, NULL, car->_penaltyTime);
				GfParmSetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, car->_bestLapTime);
				GfParmSetNum(results, path, RE_ATTR_TOP_SPEED, NULL, car->_topSpeed);
				GfParmSetNum(results, path, RE_ATTR_DAMMAGES, NULL, car->_dammage);
				GfParmSetNum(results, path, RE_ATTR_NB_PIT_STOPS, NULL, car->_nbPitStops);
			
				GfParmSetStr(results, path, RE_ATTR_MODULE, car->_modName);
				GfParmSetNum(results, path, RE_ATTR_IDX, NULL, car->_driverIndex);
			
				snprintf(path2, BUFSIZE, "%s/%s/%d", race, RM_SECT_POINTS, i + 1);
				GfParmSetNum(results, path, RE_ATTR_POINTS, NULL,
						(int)GfParmGetNum(params, path2, RE_ATTR_POINTS, NULL, 0));

				GfParmReleaseHandle(carparam);
			}
			break;
			
		case RM_TYPE_PRACTICE:
			car = s->cars[0];
			snprintf(path, BUFSIZE, "%s/%s/%s", ReInfo->track->name, RE_SECT_RESULTS, race);
			GfParmSetStr(results, path, RM_ATTR_DRVNAME, car->_name);
			break;
			
		case RM_TYPE_QUALIF:
			car = s->cars[0];
			snprintf(path, BUFSIZE, "%s/%s/%s/%s", ReInfo->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK);
			nCars = GfParmGetEltNb(results, path);
			for (i = nCars; i > 0; i--) {
				snprintf(path, BUFSIZE, "%s/%s/%s/%s/%d", ReInfo->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK, i);
				float opponentBestLapTime = GfParmGetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, 0);
			
				if (
					(car->_bestLapTime != 0.0) && 
					((round(car->_bestLapTime*1000.0f) < round(opponentBestLapTime*1000.0f)) || (opponentBestLapTime == 0.0))
				) {
					/* shift */
					snprintf(path2, BUFSIZE, "%s/%s/%s/%s/%d", ReInfo->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK, i + 1);
					GfParmSetStr(results, path2, RE_ATTR_NAME, GfParmGetStr(results, path, RE_ATTR_NAME, ""));
					GfParmSetStr(results, path2, RE_ATTR_CAR, GfParmGetStr(results, path, RE_ATTR_CAR, ""));
					GfParmSetNum(results, path2, RE_ATTR_BEST_LAP_TIME, NULL, GfParmGetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, 0));
					GfParmSetStr(results, path2, RE_ATTR_MODULE, GfParmGetStr(results, path, RM_ATTR_MODULE, ""));
					GfParmSetNum(results, path2, RE_ATTR_IDX, NULL, GfParmGetNum(results, path, RM_ATTR_IDX, NULL, 0));
					snprintf(path, BUFSIZE, "%s/%s/%d", race, RM_SECT_POINTS, i + 1);
					GfParmSetNum(results, path2, RE_ATTR_POINTS, NULL,
								(int)GfParmGetNum(params, path, RE_ATTR_POINTS, NULL, 0));
				} else {
					break;
				}
			}
			/* insert after */
			snprintf(path, BUFSIZE, "%s/%s/%s/%s/%d", ReInfo->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK, i + 1);
			GfParmSetStr(results, path, RE_ATTR_NAME, car->_name);
			
			snprintf(buf, BUFSIZE, "cars/%s/%s.xml", car->_carName, car->_carName);
			carparam = GfParmReadFile(buf, GFPARM_RMODE_STD);
			carName = GfParmGetName(carparam);
			
			GfParmSetStr(results, path, RE_ATTR_CAR, carName);
			GfParmSetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, round(car->_bestLapTime*1000.0f)/1000.0f);
			GfParmSetStr(results, path, RE_ATTR_MODULE, car->_modName);
			GfParmSetNum(results, path, RE_ATTR_IDX, NULL, car->_driverIndex);
			snprintf(path2, BUFSIZE, "%s/%s/%d", race, RM_SECT_POINTS, i + 1);
			GfParmSetNum(results, path, RE_ATTR_POINTS, NULL,
						(int)GfParmGetNum(params, path2, RE_ATTR_POINTS, NULL, 0));
		
			GfParmReleaseHandle(carparam);
			break;
	}
}


void
ReUpdateQualifCurRes(tCarElt *car)
{
	int i;
	int nCars;
	int printed;
	int maxLines;
	void *carparam;
	char *carName;
	const char *race = ReInfo->_reRaceName;
	void *results = ReInfo->results;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE], path[BUFSIZE];
	const int TIMEFMTSIZE = 256;
	char timefmt[TIMEFMTSIZE];
	
	ReResEraseScreen();
	maxLines = ReResGetLines();
	
	snprintf(buf, BUFSIZE, "%s on %s - Lap %d", car->_name, ReInfo->track->name, car->_laps);
	ReResScreenSetTitle(buf);
	
	snprintf(buf, BUFSIZE, "cars/%s/%s.xml", car->_carName, car->_carName);
	carparam = GfParmReadFile(buf, GFPARM_RMODE_STD);
	carName = GfParmGetName(carparam);
	
	printed = 0;
	snprintf(path, BUFSIZE, "%s/%s/%s/%s", ReInfo->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK);
	nCars = GfParmGetEltNb(results, path);
	nCars = MIN(nCars + 1, maxLines);
	for (i = 1; i < nCars; i++) {
		snprintf(path, BUFSIZE, "%s/%s/%s/%s/%d", ReInfo->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK, i);
		if (!printed) {
			if ((car->_bestLapTime != 0.0) && (car->_bestLapTime < GfParmGetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, 0))) {
				GfTime2Str(timefmt, TIMEFMTSIZE, car->_bestLapTime, 0);
				snprintf(buf, BUFSIZE, "%d - %s - %s (%s)", i, timefmt, car->_name, carName);
				ReResScreenSetText(buf, i - 1, 1);
				printed = 1;
			}
		}
		GfTime2Str(timefmt, TIMEFMTSIZE, GfParmGetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, 0), 0);
		snprintf(buf, BUFSIZE, "%d - %s - %s (%s)", i + printed, timefmt, GfParmGetStr(results, path, RE_ATTR_NAME, ""), GfParmGetStr(results, path, RE_ATTR_CAR, ""));
		ReResScreenSetText(buf, i - 1 + printed, 0);
	}

	if (!printed) {
		GfTime2Str(timefmt, TIMEFMTSIZE, car->_bestLapTime, 0);
		snprintf(buf, BUFSIZE, "%d - %s - %s (%s)", i, timefmt, car->_name, carName);
		ReResScreenSetText(buf, i - 1, 1);
	}

	GfParmReleaseHandle(carparam);
	ReInfo->_refreshDisplay = 1;
}

void
ReSavePracticeLap(tCarElt *car)
{
	void *results = ReInfo->results;
	tReCarInfo *info = &(ReInfo->_reCarInfo[car->index]);
	const int BUFSIZE = 1024;
	char path[BUFSIZE];

	snprintf(path, BUFSIZE, "%s/%s/%s/%d", ReInfo->track->name, RE_SECT_RESULTS, ReInfo->_reRaceName, car->_laps - 1);
	GfParmSetNum(results, path, RE_ATTR_TIME, NULL, car->_lastLapTime);
	GfParmSetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, car->_bestLapTime);
	GfParmSetNum(results, path, RE_ATTR_TOP_SPEED, NULL, info->topSpd);
	GfParmSetNum(results, path, RE_ATTR_BOT_SPEED, NULL, info->botSpd);
	GfParmSetNum(results, path, RE_ATTR_DAMMAGES, NULL, car->_dammage);    
}

int
ReDisplayResults(void)
{
	void *params = ReInfo->params;

	if (ReInfo->_displayMode != RM_DISP_MODE_CONSOLE) {
		if ((!strcmp(GfParmGetStr(params, ReInfo->_reRaceName, RM_ATTR_DISPRES, RM_VAL_YES), RM_VAL_YES)) ||
			(ReInfo->_displayMode == RM_DISP_MODE_NORMAL))
		{
			RmShowResults(ReInfo->_reGameScreen, ReInfo);
		} else {
			ReResShowCont();
		}

		return RM_ASYNC | RM_NEXT_STEP;
	}

	return RM_SYNC | RM_NEXT_STEP;
}


void
ReDisplayStandings(void)
{
	RmShowStandings(ReInfo->_reGameScreen, ReInfo);
}
