/***************************************************************************

    file                 : results.cpp
    created              : Fri Apr 14 22:36:36 CEST 2000
    copyright            : (C) 2000-2014 by Eric Espie, Bernhard Wymann
    email                : torcs@free.fr
    version              : $Id: results.cpp,v 1.6.2.10 2014/05/23 08:38:31 berniw Exp $

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
    Display results.
    @author Bernhard Wymann, Eric Espie
    @version $Id: results.cpp,v 1.6.2.10 2014/05/23 08:38:31 berniw Exp $
*/

#include <stdlib.h>
#include <stdio.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <tgfclient.h>
#include <osspec.h>
#include <racescreens.h>
#include <robottools.h>
#include <robot.h>
#include <portability.h>

static int	rmSaveId;
static void	*rmScrHdle = NULL;

static void rmPracticeResults(void *prevHdle, tRmInfo *info, int start);
static void rmRaceResults(void *prevHdle, tRmInfo *info, int start);
static void rmQualifResults(void *prevHdle, tRmInfo *info, int start);
static void rmShowStandings(void *prevHdle, tRmInfo *info, int start);

#define MAX_LINES	20

typedef struct
{
	void *prevHdle;
	tRmInfo *info;
	int start;
} tRaceCall;

tRaceCall RmNextRace;
tRaceCall RmPrevRace;


static void rmSaveRes(void *vInfo)
{
	tRmInfo *info = (tRmInfo *)vInfo;
	GfParmCreateDirectory(0, info->results);
	GfParmWriteFile(0, info->results, "Results");
	GfuiVisibilitySet(rmScrHdle, rmSaveId, GFUI_INVISIBLE);
}


static void rmChgPracticeScreen(void *vprc)
{
	void *prevScr = rmScrHdle;
	tRaceCall *prc = (tRaceCall*)vprc;

	rmPracticeResults(prc->prevHdle, prc->info, prc->start);
	GfuiScreenRelease(prevScr);
}


static void rmPracticeResults(void *prevHdle, tRmInfo *info, int start)
{
	void *results = info->results;
	const char *race = info->_reRaceName;
	int i;
	int x1, x2, x3, x4, x5, x6;
	int offset;
	int y;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE];
	char path[BUFSIZE];
	const int TIMEFMTSIZE = 256;
	char timefmt[TIMEFMTSIZE];
	float fgcolor[4] = {1.0, 0.0, 1.0, 1.0};
	int totLaps;

	rmScrHdle = GfuiScreenCreate();
	snprintf(buf, BUFSIZE, "Practice Results");
	GfuiTitleCreate(rmScrHdle, buf, strlen(buf));
	snprintf(path, BUFSIZE, "%s/%s/%s", info->track->name, RE_SECT_RESULTS, race);
	snprintf(buf, BUFSIZE, "%s on track %s", GfParmGetStr(results, path, RM_ATTR_DRVNAME, ""), info->track->name);
	GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_LARGE_C,
			320, 420, GFUI_ALIGN_HC_VB, 0);
	GfuiScreenAddBgImg(rmScrHdle, "data/img/splash-result.png");
	
	offset = 90;
	
	x1 = offset + 30;
	x2 = offset + 50;
	x3 = offset + 130;
	x4 = offset + 240;
	x5 = offset + 310;
	x6 = offset + 400;
	
	y = 400;
	GfuiLabelCreateEx(rmScrHdle, "Lap",       fgcolor, GFUI_FONT_MEDIUM_C, x1, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Time",      fgcolor, GFUI_FONT_MEDIUM_C, x2+20, y, GFUI_ALIGN_HL_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Best",      fgcolor, GFUI_FONT_MEDIUM_C, x3+20, y, GFUI_ALIGN_HL_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Top Spd",   fgcolor, GFUI_FONT_MEDIUM_C, x4, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Min Spd",   fgcolor, GFUI_FONT_MEDIUM_C, x5, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Damages",  fgcolor, GFUI_FONT_MEDIUM_C, x6, y, GFUI_ALIGN_HC_VB, 0);
	y -= 20;
	
	snprintf(path, BUFSIZE, "%s/%s/%s", info->track->name, RE_SECT_RESULTS, race);
	totLaps = (int)GfParmGetEltNb(results, path);
	for (i = 0 + start; i < MIN(start + MAX_LINES, totLaps); i++) {
		snprintf(path, BUFSIZE, "%s/%s/%s/%d", info->track->name, RE_SECT_RESULTS, race, i + 1);

		/* Lap */
		snprintf(buf, BUFSIZE, "%d", i+1);
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C, x1, y, GFUI_ALIGN_HC_VB, 0);

		/* Time */
		GfTime2Str(timefmt, TIMEFMTSIZE, GfParmGetNum(results, path, RE_ATTR_TIME, NULL, 0), 0);
		GfuiLabelCreate(rmScrHdle, timefmt, GFUI_FONT_MEDIUM_C, x2, y, GFUI_ALIGN_HL_VB, 0);

		/* Best Lap Time */
		GfTime2Str(timefmt, TIMEFMTSIZE, GfParmGetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, 0), 0);
		GfuiLabelCreate(rmScrHdle, timefmt, GFUI_FONT_MEDIUM_C, x3, y, GFUI_ALIGN_HL_VB, 0);

		/* Top Spd */
		snprintf(buf, BUFSIZE, "%d", (int)(GfParmGetNum(results, path, RE_ATTR_TOP_SPEED, NULL, 0) * 3.6));
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C, x4, y, GFUI_ALIGN_HC_VB, 0);

		/* Min Spd */
		snprintf(buf, BUFSIZE, "%d", (int)(GfParmGetNum(results, path, RE_ATTR_BOT_SPEED, NULL, 0) * 3.6));
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C, x5, y, GFUI_ALIGN_HC_VB, 0);

		/* Damages */
		snprintf(buf, BUFSIZE, "%d", (int)(GfParmGetNum(results, path, RE_ATTR_DAMMAGES, NULL, 0)));
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C, x6, y, GFUI_ALIGN_HC_VB, 0);

		y -= 15;
	}

	if (start > 0) {
		RmPrevRace.prevHdle = prevHdle;
		RmPrevRace.info     = info;
		RmPrevRace.start    = start - MAX_LINES;
		GfuiGrButtonCreate(rmScrHdle, "data/img/arrow-up.png", "data/img/arrow-up.png",
				"data/img/arrow-up.png", "data/img/arrow-up-pushed.png",
				80, 40, GFUI_ALIGN_HL_VB, 1,
				(void*)&RmPrevRace, rmChgPracticeScreen,
				NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
		GfuiAddSKey(rmScrHdle, GLUT_KEY_PAGE_UP,   "Previous Results", (void*)&RmPrevRace, rmChgPracticeScreen, NULL);
	}
	
	GfuiButtonCreate(rmScrHdle,
			"Continue",
			GFUI_FONT_LARGE,
			320,
			40,
			150,
			GFUI_ALIGN_HC_VB,
			0,
			prevHdle,
			GfuiScreenReplace,
			NULL,
			(tfuiCallback)NULL,
			(tfuiCallback)NULL);

	if (i < totLaps) {
		RmNextRace.prevHdle = prevHdle;
		RmNextRace.info     = info;
		RmNextRace.start    = start + MAX_LINES;
		GfuiGrButtonCreate(rmScrHdle, "data/img/arrow-down.png", "data/img/arrow-down.png",
				"data/img/arrow-down.png", "data/img/arrow-down-pushed.png",
				540, 40, GFUI_ALIGN_HL_VB, 1,
				(void*)&RmNextRace, rmChgPracticeScreen,
				NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
		GfuiAddSKey(rmScrHdle, GLUT_KEY_PAGE_DOWN, "Next Results", (void*)&RmNextRace, rmChgPracticeScreen, NULL);
	}

	GfuiAddKey(rmScrHdle, (unsigned char)27, "", prevHdle, GfuiScreenReplace, NULL);
	GfuiAddKey(rmScrHdle, (unsigned char)13, "", prevHdle, GfuiScreenReplace, NULL);
	GfuiAddSKey(rmScrHdle, GLUT_KEY_F12, "Take a Screen Shot", NULL, GfuiScreenShot, NULL);

	GfuiScreenActivate(rmScrHdle);
}


static void rmChgRaceScreen(void *vprc)
{
	void *prevScr = rmScrHdle;
	tRaceCall *prc = (tRaceCall*)vprc;

	rmRaceResults(prc->prevHdle, prc->info, prc->start);
	GfuiScreenRelease(prevScr);
}


static void rmRaceResults(void *prevHdle, tRmInfo *info, int start)
{
	void *results = info->results;
	const char *race = info->_reRaceName;
	int i;
	int x1, x2, x3, x4, x5, x6, x7, x8, x9;
	int dlap;
	int y;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE];
	char path[BUFSIZE];
	const int TIMEFMTSIZE = 256;
	char timefmt[TIMEFMTSIZE];
	float fgcolor[4] = {1.0, 0.0, 1.0, 1.0};
	int laps, totLaps;
	tdble refTime;
	int nbCars;

	rmScrHdle = GfuiScreenCreate();
	snprintf(buf, BUFSIZE, "Race Results");
	GfuiTitleCreate(rmScrHdle, buf, strlen(buf));
	snprintf(buf, BUFSIZE, "%s", info->track->name);
	GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_LARGE_C,
			320, 420, GFUI_ALIGN_HC_VB, 0);
	GfuiScreenAddBgImg(rmScrHdle, "data/img/splash-result.png");
	
	x1 = 30;
	x2 = 60;
	x3 = 260;
	x4 = 330;
	x5 = 360;
	x6 = 420;
	x7 = 490;
	x8 = 545;
	x9 = 630;
	
	y = 400;
	GfuiLabelCreateEx(rmScrHdle, "Rank",      fgcolor, GFUI_FONT_MEDIUM_C, x1, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Driver",    fgcolor, GFUI_FONT_MEDIUM_C, x2+10, y, GFUI_ALIGN_HL_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Total",     fgcolor, GFUI_FONT_MEDIUM_C, x3, y, GFUI_ALIGN_HR_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Best",      fgcolor, GFUI_FONT_MEDIUM_C, x4, y, GFUI_ALIGN_HR_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Laps",      fgcolor, GFUI_FONT_MEDIUM_C, x5, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Top Spd",   fgcolor, GFUI_FONT_MEDIUM_C, x6, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Damage",    fgcolor, GFUI_FONT_MEDIUM_C, x7, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Pit",       fgcolor, GFUI_FONT_MEDIUM_C, x8, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Penalty",   fgcolor, GFUI_FONT_MEDIUM_C, x9, y, GFUI_ALIGN_HR_VB, 0);	
	y -= 20;
	
	snprintf(path, BUFSIZE, "%s/%s/%s", info->track->name, RE_SECT_RESULTS, race);
	totLaps = (int)GfParmGetNum(results, path, RE_ATTR_LAPS, NULL, 0);
	snprintf(path, BUFSIZE, "%s/%s/%s/%s/%d", info->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK, 1);
	refTime = GfParmGetNum(results, path, RE_ATTR_TIME, NULL, 0);
	snprintf(path, BUFSIZE, "%s/%s/%s/%s", info->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK);
	nbCars = (int)GfParmGetEltNb(results, path);
	for (i = start; i < MIN(start + MAX_LINES, nbCars); i++) {
		snprintf(path, BUFSIZE, "%s/%s/%s/%s/%d", info->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK, i + 1);
		laps = (int)GfParmGetNum(results, path, RE_ATTR_LAPS, NULL, 0);

		snprintf(buf, BUFSIZE, "%d", i+1);
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C,
				x1, y, GFUI_ALIGN_HC_VB, 0);

		GfuiLabelCreate(rmScrHdle, GfParmGetStr(results, path, RE_ATTR_NAME, ""), GFUI_FONT_MEDIUM_C,
				x2, y, GFUI_ALIGN_HL_VB, 0);

		if (laps == totLaps) {
			if (i == 0) {
				GfTime2Str(timefmt, TIMEFMTSIZE, GfParmGetNum(results, path, RE_ATTR_TIME, NULL, 0), 0);
			} else {
				GfTime2Str(timefmt, TIMEFMTSIZE, GfParmGetNum(results, path, RE_ATTR_TIME, NULL, 0) - refTime, 1);
			}
			GfuiLabelCreate(rmScrHdle, timefmt, GFUI_FONT_MEDIUM_C, x3, y, GFUI_ALIGN_HR_VB, 0);
		} else {
			dlap = totLaps - laps;
			if (dlap == 1) {
				snprintf(buf, BUFSIZE, "+1 Lap");
			} else {
				snprintf(buf, BUFSIZE, "+%d Laps", dlap);
			}
			GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C, x3, y, GFUI_ALIGN_HR_VB, 0);

		}

		GfTime2Str(timefmt, TIMEFMTSIZE, GfParmGetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, 0), 0);
		GfuiLabelCreate(rmScrHdle, timefmt, GFUI_FONT_MEDIUM_C,
				x4, y, GFUI_ALIGN_HR_VB, 0);

		snprintf(buf, BUFSIZE, "%d", laps);
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C,
				x5, y, GFUI_ALIGN_HC_VB, 0);

		snprintf(buf, BUFSIZE, "%d", (int)(GfParmGetNum(results, path, RE_ATTR_TOP_SPEED, NULL, 0) * 3.6));
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C,
				x6, y, GFUI_ALIGN_HC_VB, 0);

		snprintf(buf, BUFSIZE, "%d", (int)(GfParmGetNum(results, path, RE_ATTR_DAMMAGES, NULL, 0)));
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C,
				x7, y, GFUI_ALIGN_HC_VB, 0);

		snprintf(buf, BUFSIZE, "%d", (int)(GfParmGetNum(results, path, RE_ATTR_NB_PIT_STOPS, NULL, 0)));
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C,
				x8, y, GFUI_ALIGN_HC_VB, 0);

		GfTime2Str(timefmt, TIMEFMTSIZE, GfParmGetNum(results, path, RE_ATTR_PENALTYTIME, NULL, 0), 0);
		GfuiLabelCreate(rmScrHdle, timefmt, GFUI_FONT_MEDIUM_C, x9, y, GFUI_ALIGN_HR_VB, 0);

		y -= 15;
	}

	if (start > 0) {
		RmPrevRace.prevHdle = prevHdle;
		RmPrevRace.info     = info;
		RmPrevRace.start    = start - MAX_LINES;
		GfuiGrButtonCreate(rmScrHdle, "data/img/arrow-up.png", "data/img/arrow-up.png",
				"data/img/arrow-up.png", "data/img/arrow-up-pushed.png",
				80, 40, GFUI_ALIGN_HL_VB, 1,
				(void*)&RmPrevRace, rmChgRaceScreen,
				NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
		GfuiAddSKey(rmScrHdle, GLUT_KEY_PAGE_UP,   "Previous Results", (void*)&RmPrevRace, rmChgRaceScreen, NULL);
	}

	GfuiButtonCreate(rmScrHdle,
			"Continue",
			GFUI_FONT_LARGE,
			/* 210, */
			320,
			40,
			150,
			GFUI_ALIGN_HC_VB,
			0,
			prevHdle,
			GfuiScreenReplace,
			NULL,
			(tfuiCallback)NULL,
			(tfuiCallback)NULL);

	if (i < nbCars) {
		RmNextRace.prevHdle = prevHdle;
		RmNextRace.info     = info;
		RmNextRace.start    = start + MAX_LINES;
		GfuiGrButtonCreate(rmScrHdle, "data/img/arrow-down.png", "data/img/arrow-down.png",
				"data/img/arrow-down.png", "data/img/arrow-down-pushed.png",
				540, 40, GFUI_ALIGN_HL_VB, 1,
				(void*)&RmNextRace, rmChgRaceScreen,
				NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
		GfuiAddSKey(rmScrHdle, GLUT_KEY_PAGE_DOWN, "Next Results", (void*)&RmNextRace, rmChgRaceScreen, NULL);
	}

	GfuiAddKey(rmScrHdle, (unsigned char)27, "", prevHdle, GfuiScreenReplace, NULL);
	GfuiAddKey(rmScrHdle, (unsigned char)13, "", prevHdle, GfuiScreenReplace, NULL);
	GfuiAddSKey(rmScrHdle, GLUT_KEY_F12, "Take a Screen Shot", NULL, GfuiScreenShot, NULL);

	GfuiScreenActivate(rmScrHdle);
}


static void rmChgQualifScreen(void *vprc)
{
	void *prevScr = rmScrHdle;
	tRaceCall *prc = (tRaceCall*)vprc;

	rmQualifResults(prc->prevHdle, prc->info, prc->start);
	GfuiScreenRelease(prevScr);
}


static void rmQualifResults(void *prevHdle, tRmInfo *info, int start)
{
	void *results = info->results;
	const char *race = info->_reRaceName;
	int i;
	int x1, x2, x3;
	int y;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE];
	char path[BUFSIZE];
	const int TIMEFMTSIZE = 256;
	char timefmt[TIMEFMTSIZE];
	float fgcolor[4] = {1.0, 0.0, 1.0, 1.0};
	int nbCars;
	int offset;

	rmScrHdle = GfuiScreenCreate();
	snprintf(buf, BUFSIZE, "Qualification Results");
	GfuiTitleCreate(rmScrHdle, buf, strlen(buf));
	snprintf(buf, BUFSIZE, "%s", info->track->name);
	GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_LARGE_C,
			320, 420, GFUI_ALIGN_HC_VB, 0);
	GfuiScreenAddBgImg(rmScrHdle, "data/img/splash-result.png");

	offset = 200;
	x1 = offset + 30;
	x2 = offset + 60;
	x3 = offset + 240;
	
	y = 400;
	GfuiLabelCreateEx(rmScrHdle, "Rank",      fgcolor, GFUI_FONT_MEDIUM_C, x1, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Driver",    fgcolor, GFUI_FONT_MEDIUM_C, x2+10, y, GFUI_ALIGN_HL_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Time",      fgcolor, GFUI_FONT_MEDIUM_C, x3, y, GFUI_ALIGN_HR_VB, 0);
	y -= 20;
	
	snprintf(path, BUFSIZE, "%s/%s/%s/%s", info->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK);
	nbCars = (int)GfParmGetEltNb(results, path);
	
	for (i = start; i < MIN(start + MAX_LINES, nbCars); i++) {
		snprintf(path, BUFSIZE, "%s/%s/%s/%s/%d", info->track->name, RE_SECT_RESULTS, race, RE_SECT_RANK, i + 1);

		snprintf(buf, BUFSIZE, "%d", i+1);
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C,
				x1, y, GFUI_ALIGN_HC_VB, 0);

		GfuiLabelCreate(rmScrHdle, GfParmGetStr(results, path, RE_ATTR_NAME, ""), GFUI_FONT_MEDIUM_C,
				x2, y, GFUI_ALIGN_HL_VB, 0);

		GfTime2Str(timefmt, TIMEFMTSIZE, GfParmGetNum(results, path, RE_ATTR_BEST_LAP_TIME, NULL, 0), 0);
		GfuiLabelCreate(rmScrHdle, timefmt, GFUI_FONT_MEDIUM_C,
				x3, y, GFUI_ALIGN_HR_VB, 0);
		y -= 15;
	}


	if (start > 0) {
		RmPrevRace.prevHdle = prevHdle;
		RmPrevRace.info     = info;
		RmPrevRace.start    = start - MAX_LINES;
		GfuiGrButtonCreate(rmScrHdle, "data/img/arrow-up.png", "data/img/arrow-up.png",
				"data/img/arrow-up.png", "data/img/arrow-up-pushed.png",
				80, 40, GFUI_ALIGN_HL_VB, 1,
				(void*)&RmPrevRace, rmChgQualifScreen,
				NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
		GfuiAddSKey(rmScrHdle, GLUT_KEY_PAGE_UP,   "Previous Results", (void*)&RmPrevRace, rmChgQualifScreen, NULL);
	}

	GfuiButtonCreate(rmScrHdle,
			"Continue",
			GFUI_FONT_LARGE,
			320,
			40,
			150,
			GFUI_ALIGN_HC_VB,
			0,
			prevHdle,
			GfuiScreenReplace,
			NULL,
			(tfuiCallback)NULL,
			(tfuiCallback)NULL);

	if (i < nbCars) {
		RmNextRace.prevHdle = prevHdle;
		RmNextRace.info     = info;
		RmNextRace.start    = start + MAX_LINES;
		GfuiGrButtonCreate(rmScrHdle, "data/img/arrow-down.png", "data/img/arrow-down.png",
				"data/img/arrow-down.png", "data/img/arrow-down-pushed.png",
				540, 40, GFUI_ALIGN_HL_VB, 1,
				(void*)&RmNextRace, rmChgQualifScreen,
				NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
		GfuiAddSKey(rmScrHdle, GLUT_KEY_PAGE_DOWN, "Next Results", (void*)&RmNextRace, rmChgQualifScreen, NULL);
	}

	GfuiAddKey(rmScrHdle, (unsigned char)27, "", prevHdle, GfuiScreenReplace, NULL);
	GfuiAddKey(rmScrHdle, (unsigned char)13, "", prevHdle, GfuiScreenReplace, NULL);
	GfuiAddSKey(rmScrHdle, GLUT_KEY_F12, "Take a Screen Shot", NULL, GfuiScreenShot, NULL);

	GfuiScreenActivate(rmScrHdle);
}


static void rmChgStandingScreen(void *vprc)
{
	void *prevScr = rmScrHdle;
	tRaceCall *prc = (tRaceCall*)vprc;

	rmShowStandings(prc->prevHdle, prc->info, prc->start);
	GfuiScreenRelease(prevScr);
}


static void rmShowStandings(void *prevHdle, tRmInfo *info, int start)
{
	int i;
	int x1, x2, x3;
	int y;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE];
	char path[BUFSIZE];
	float fgcolor[4] = {1.0, 0.0, 1.0, 1.0};
	int nbCars;
	int offset;
	void *results = info->results;
	const char *race = info->_reRaceName;
	
	rmScrHdle = GfuiScreenCreate();
	snprintf(buf, BUFSIZE, "%s Results", race);
	GfuiTitleCreate(rmScrHdle, buf, strlen(buf));
	
	GfuiScreenAddBgImg(rmScrHdle, "data/img/splash-result.png");
	
	offset = 200;
	x1 = offset + 30;
	x2 = offset + 60;
	x3 = offset + 240;
	
	y = 400;
	GfuiLabelCreateEx(rmScrHdle, "Rank",      fgcolor, GFUI_FONT_MEDIUM_C, x1, y, GFUI_ALIGN_HC_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Driver",    fgcolor, GFUI_FONT_MEDIUM_C, x2+10, y, GFUI_ALIGN_HL_VB, 0);
	GfuiLabelCreateEx(rmScrHdle, "Points",      fgcolor, GFUI_FONT_MEDIUM_C, x3, y, GFUI_ALIGN_HR_VB, 0);
	y -= 20;
	
	nbCars = (int)GfParmGetEltNb(results, RE_SECT_STANDINGS);
	for (i = start; i < MIN(start + MAX_LINES, nbCars); i++) {
		snprintf(path, BUFSIZE, "%s/%d", RE_SECT_STANDINGS, i + 1);
		
		snprintf(buf, BUFSIZE, "%d", i+1);
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C,
				x1, y, GFUI_ALIGN_HC_VB, 0);
		
		GfuiLabelCreate(rmScrHdle, GfParmGetStr(results, path, RE_ATTR_NAME, ""), GFUI_FONT_MEDIUM_C,
				x2, y, GFUI_ALIGN_HL_VB, 0);
		
		snprintf(buf, BUFSIZE, "%d", (int)GfParmGetNum(results, path, RE_ATTR_POINTS, NULL, 0));
		GfuiLabelCreate(rmScrHdle, buf, GFUI_FONT_MEDIUM_C,
				x3, y, GFUI_ALIGN_HR_VB, 0);
		y -= 15;
	}
	
	
	if (start > 0) {
		RmPrevRace.prevHdle = prevHdle;
		RmPrevRace.info     = info;
		RmPrevRace.start    = start - MAX_LINES;
		GfuiGrButtonCreate(rmScrHdle, "data/img/arrow-up.png", "data/img/arrow-up.png",
					"data/img/arrow-up.png", "data/img/arrow-up-pushed.png",
					80, 40, GFUI_ALIGN_HL_VB, 1,
					(void*)&RmPrevRace, rmChgStandingScreen,
					NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
		GfuiAddSKey(rmScrHdle, GLUT_KEY_PAGE_UP,   "Previous Results", (void*)&RmPrevRace, rmChgStandingScreen, NULL);
	}
	
	GfuiButtonCreate(rmScrHdle,
				"Continue",
				GFUI_FONT_LARGE,
				210,
				40,
				150,
				GFUI_ALIGN_HC_VB,
				0,
				prevHdle,
				GfuiScreenReplace,
				NULL,
				(tfuiCallback)NULL,
				(tfuiCallback)NULL);
	
	rmSaveId = GfuiButtonCreate(rmScrHdle,
				"Save",
				GFUI_FONT_LARGE,
				430,
				40,
				150,
				GFUI_ALIGN_HC_VB,
				0,
				info,
				rmSaveRes,
				NULL,
				(tfuiCallback)NULL,
				(tfuiCallback)NULL);
	
	if (i < nbCars) {
		RmNextRace.prevHdle = prevHdle;
		RmNextRace.info     = info;
		RmNextRace.start    = start + MAX_LINES;
		GfuiGrButtonCreate(rmScrHdle, "data/img/arrow-down.png", "data/img/arrow-down.png",
					"data/img/arrow-down.png", "data/img/arrow-down-pushed.png",
					540, 40, GFUI_ALIGN_HL_VB, 1,
					(void*)&RmNextRace, rmChgStandingScreen,
					NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
		GfuiAddSKey(rmScrHdle, GLUT_KEY_PAGE_DOWN, "Next Results", (void*)&RmNextRace, rmChgStandingScreen, NULL);
	}
	
	GfuiAddKey(rmScrHdle, (unsigned char)27, "", prevHdle, GfuiScreenReplace, NULL);
	GfuiAddKey(rmScrHdle, (unsigned char)13, "", prevHdle, GfuiScreenReplace, NULL);
	GfuiAddSKey(rmScrHdle, GLUT_KEY_F12, "Take a Screen Shot", NULL, GfuiScreenShot, NULL);
	
	GfuiScreenActivate(rmScrHdle);
}


/** @brief Display results
 *  @ingroup racemantools
 *  @param[in] prevHdle Handle to previous result screen (used if the results require more than one screen)
 *  @param[in] info tRmInfo.results carries the result parameter set handle
 */
void RmShowResults(void *prevHdle, tRmInfo *info)
{
	switch (info->s->_raceType) {
		case RM_TYPE_PRACTICE:
			rmPracticeResults(prevHdle, info, 0);
			return;

		case RM_TYPE_RACE:
			rmRaceResults(prevHdle, info, 0);
			return;

		case RM_TYPE_QUALIF:
			rmQualifResults(prevHdle, info, 0);
			return;
	}
}


/** @brief Display standings
 *  @ingroup racemantools
 *  @param[in] prevHdle Handle to previous standings screen (used if the standings require more than one screen)
 *  @param[in] info tRmInfo.results carries the result parameter set handle containing the standings
 */
void RmShowStandings(void *prevHdle, tRmInfo *info)
{
    rmShowStandings(prevHdle, info, 0);
}
