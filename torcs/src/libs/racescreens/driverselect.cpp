/***************************************************************************
                  driverselect.cpp -- drivers interactive selection                              
                             -------------------                                         
    created              : Mon Aug 16 20:40:44 CEST 1999
    copyright            : (C) 1999-2014 by Eric Espie, Bernhard Wymann                         
    email                : torcs@free.fr   
    version              : $Id: driverselect.cpp,v 1.5.2.6 2014/05/20 12:16:46 berniw Exp $                                  
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
    Driver selection screen.
    @author Bernhard Wymann, Eric Espie
    @version $Id: driverselect.cpp,v 1.5.2.6 2014/05/20 12:16:46 berniw Exp $
*/


#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <tgfclient.h>
#include <track.h>
#include <car.h>
#include <raceman.h>
#include <robot.h>
#include <racescreens.h>
#include <portability.h>

static void *scrHandle;
static tRmDrvSelect	*ds;
static int selectedScrollList, unselectedScrollList;
static int FocDrvLabelId;
static int PickDrvNameLabelId;
static int PickDrvCarLabelId;
static int PickDrvCategoryLabelId;
static float aColor[] = { 1.0, 0.0, 0.0, 1.0 };
static int nbSelectedDrivers;
static int nbMaxSelectedDrivers;

typedef struct DrvElt
{
	int index;
	char *dname;
	char *name;
	int sel;
	int human;
	void *car;
	GF_TAILQ_ENTRY(struct DrvElt)	link;
} tDrvElt;

GF_TAILQ_HEAD(DrvListHead, tDrvElt);

tDrvListHead DrvList;

static void rmFreeDrvList(void);


static void rmdsActivate(void * /* notused */)
{
    /* call display function of graphic */
}


static void rmdsDeactivate(void *screen)
{
	rmFreeDrvList();    
	GfuiScreenRelease(scrHandle);

	if (screen) {
		GfuiScreenActivate(screen);
	}
}


static void rmdsSetFocus(void * /* dummy */)
{
	char *name;
	tDrvElt	*curDrv;

	name = GfuiScrollListGetSelectedElement(scrHandle, selectedScrollList, (void**)&curDrv);
	if (name) {
		GfParmSetStr(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSED, curDrv->dname);
		GfParmSetNum(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSEDIDX, (char*)NULL, curDrv->index);
		GfuiLabelSetText(scrHandle, FocDrvLabelId, curDrv->name);
	}
}


static void rmdsSelect(void * /* dummy */)
{
	char *name;
	tDrvElt	*curDrv;
	int	index;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE];

	snprintf(buf, BUFSIZE, "%s", RM_SECT_DRIVERS);
	GfParmListClean(ds->param, buf);
	name = GfuiScrollListExtractElement(scrHandle, selectedScrollList, 0, (void**)&curDrv);
	index = 1;
	
	while (name != NULL) {
		snprintf(buf, BUFSIZE, "%s/%d", RM_SECT_DRIVERS, index);
		GfParmSetNum(ds->param, buf, RM_ATTR_IDX, (char*)NULL, curDrv->index);
		GfParmSetStr(ds->param, buf, RM_ATTR_MODULE, curDrv->dname);
		index++;
		name = GfuiScrollListExtractElement(scrHandle, selectedScrollList, 0, (void**)&curDrv);
	}
	
	rmdsDeactivate(ds->nextScreen);
}


static void rmMove(void *vd)
{
	GfuiScrollListMoveSelectedElement(scrHandle, selectedScrollList, (long)vd);
	GfuiScrollListMoveSelectedElement(scrHandle, unselectedScrollList, (long)vd);
}


static void rmdsClickOnDriver(void * /* dummy */)
{
	char *name;
	tDrvElt	*curDrv;
	void *robhdle;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE];

	name = GfuiScrollListGetSelectedElement(scrHandle, selectedScrollList, (void**)&curDrv);
	if (!name) {
		name = GfuiScrollListGetSelectedElement(scrHandle, unselectedScrollList, (void**)&curDrv);
	}

	if (name) {
		GfuiLabelSetText(scrHandle, PickDrvNameLabelId, curDrv->name);
		/* search driver infos */
		snprintf(buf, BUFSIZE, "%sdrivers/%s/%s.xml", GetLocalDir(), curDrv->dname, curDrv->dname);
		robhdle = GfParmReadFile(buf, GFPARM_RMODE_STD);
		
		if (!robhdle) {
			snprintf(buf, BUFSIZE, "drivers/%s/%s.xml", curDrv->dname, curDrv->dname);
			robhdle = GfParmReadFile(buf, GFPARM_RMODE_STD);
		}
		
		if (robhdle != NULL) {
			snprintf(buf, BUFSIZE, "%s/%s/%d", ROB_SECT_ROBOTS, ROB_LIST_INDEX, curDrv->index);
			GfuiLabelSetText(scrHandle, PickDrvCarLabelId, GfParmGetName(curDrv->car));
			GfuiLabelSetText(scrHandle, PickDrvCategoryLabelId, GfParmGetStr(curDrv->car, SECT_CAR, PRM_CATEGORY, ""));
			GfParmReleaseHandle(robhdle);
		}
	}
}

static void rmSelectDeselect(void * /* dummy */ )
{
	char *name;
	int src, dst;
	tDrvElt *curDrv;
	int robotIdx;
	int sel;
	
	sel = 0;
	name = 0;
	if (nbSelectedDrivers < nbMaxSelectedDrivers) {
		src = unselectedScrollList;
		name = GfuiScrollListExtractSelectedElement(scrHandle, src, (void**)&curDrv);
		if (name) {
			dst = selectedScrollList;
			GfuiScrollListInsertElement(scrHandle, dst, name, 100, (void*)curDrv);
			nbSelectedDrivers++;
		}
	}

	if (!name) {
		sel = 1;
		src = selectedScrollList;
		name = GfuiScrollListExtractSelectedElement(scrHandle, src, (void**)&curDrv);
		if (name) {
			dst = unselectedScrollList;
			if (curDrv->human) {
				GfuiScrollListInsertElement(scrHandle, dst, name, 0, (void*)curDrv);
			} else {
				GfuiScrollListInsertElement(scrHandle, dst, name, 100, (void*)curDrv);
			}
			nbSelectedDrivers--;
		} else {
			return;
		}
	}
	
	const char* cardllname = GfParmGetStr(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSED, "");
	robotIdx = (int)GfParmGetNum(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSEDIDX, (char*)NULL, 0);
	if (sel) {
		cardllname = GfParmGetStr(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSED, "");
		robotIdx = (int)GfParmGetNum(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSEDIDX, (char*)NULL, 0);
		if ((curDrv->index == robotIdx) && (strcmp(curDrv->dname, cardllname) == 0)) {
			/* the focused element was deselected select a new one */
			name = GfuiScrollListGetElement(scrHandle, selectedScrollList, 0, (void**)&curDrv);
			if (name) {
				GfParmSetStr(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSED, curDrv->dname);
				GfParmSetNum(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSEDIDX, (char*)NULL, curDrv->index);
				GfuiLabelSetText(scrHandle, FocDrvLabelId, curDrv->name);
			} else {
				GfParmSetStr(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSED, "");
				GfParmSetNum(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSEDIDX, (char*)NULL, 0);
				GfuiLabelSetText(scrHandle, FocDrvLabelId, "");
			}
		}
	} else {
		if ((strlen(cardllname) == 0) || (curDrv->human)) {
			GfParmSetStr(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSED, curDrv->dname);
			GfParmSetNum(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSEDIDX, (char*)NULL, curDrv->index);
			GfuiLabelSetText(scrHandle, FocDrvLabelId, curDrv->name);
		}
	}
	rmdsClickOnDriver(NULL);
}


static void rmdsAddKeys(void)
{
	GfuiAddKey(scrHandle, 27, "Cancel Selection", ds->prevScreen, rmdsDeactivate, NULL);
	GfuiAddKey(scrHandle, 13, "Accept Selection", NULL, rmdsSelect, NULL);
	GfuiAddSKey(scrHandle, GLUT_KEY_F1, "Help", scrHandle, GfuiHelpScreen, NULL);
	GfuiAddSKey(scrHandle, GLUT_KEY_F12, "Screen-Shot", NULL, GfuiScreenShot, NULL);
	GfuiAddKey(scrHandle, '-', "Move Up", (void*)-1, rmMove, NULL);
	GfuiAddKey(scrHandle, '+', "Move Down", (void*)1, rmMove, NULL);
	GfuiAddKey(scrHandle, ' ', "Select/Deselect", NULL, rmSelectDeselect, NULL);
	GfuiAddKey(scrHandle, 'f', "Set Focus", NULL, rmdsSetFocus, NULL);    
}


/** @brief Drivers list selection, the race manager parameter set is handed over in vs, tRmDrvSelect.param
 *  @ingroup racemantools
 *  @param[in,out] vs Pointer on tRmDrvSelect structure (cast to void)
 *  @note The race manager parameter set is modified in memory but not persisted.
 */
void RmDriversSelect(void *vs)
{
	tModList *list;
	tModList *curmod;
	char *sp;
	int i, index;
	tDrvElt	*curDrv;
	int	nCars, robotIdx;
	void *robhdle;
	struct stat st;
	void *carhdle;
	int	 human;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE];
	char path[BUFSIZE];
	char dname[BUFSIZE];

#define B_BASE  380
#define B_HT    30

	ds = (tRmDrvSelect*)vs;
	
	GF_TAILQ_INIT(&DrvList);
	
	scrHandle = GfuiScreenCreateEx((float*)NULL, NULL, rmdsActivate, NULL, (tfuiCallback)NULL, 1);
	GfuiScreenAddBgImg(scrHandle, "data/img/splash-qrdrv.png");
	
	GfuiTitleCreate(scrHandle, "Select Drivers", sizeof("Select Drivers"));
	
	GfuiLabelCreate(scrHandle,
			"Selected",
			GFUI_FONT_LARGE,
			120, 400, GFUI_ALIGN_HC_VB,
			0);
	
	GfuiLabelCreate(scrHandle,
			"Not Selected",
			GFUI_FONT_LARGE,
			496, 400, GFUI_ALIGN_HC_VB,
			0);
	
	selectedScrollList = GfuiScrollListCreate(scrHandle, GFUI_FONT_MEDIUM_C, 20, 80, GFUI_ALIGN_HL_VB,
							200, 310, GFUI_SB_RIGHT, NULL, rmdsClickOnDriver);
	unselectedScrollList = GfuiScrollListCreate(scrHandle, GFUI_FONT_MEDIUM_C, 396, 80, GFUI_ALIGN_HL_VB,
						200, 310, GFUI_SB_RIGHT, NULL, rmdsClickOnDriver);
	
	
	GfuiButtonCreate(scrHandle, "Accept", GFUI_FONT_LARGE, 210, 40, 150, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				NULL, rmdsSelect, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
	
	GfuiButtonCreate(scrHandle, "Cancel", GFUI_FONT_LARGE, 430, 40, 150, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				ds->prevScreen, rmdsDeactivate, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
	
	GfuiButtonCreate(scrHandle, "Move Up", GFUI_FONT_MEDIUM, 320, B_BASE, 100, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				(void*)-1, rmMove, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
	
	GfuiButtonCreate(scrHandle, "Move Down", GFUI_FONT_MEDIUM, 320, B_BASE - B_HT, 100, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				(void*)1, rmMove, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
	
	GfuiButtonCreate(scrHandle, "(De)Select", GFUI_FONT_MEDIUM, 320, B_BASE - 2 * B_HT, 100, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				(void*)0, rmSelectDeselect, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
	
	GfuiButtonCreate(scrHandle, "Set Focus", GFUI_FONT_MEDIUM, 320, B_BASE - 3 * B_HT, 100, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				NULL, rmdsSetFocus, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);
	
	list = (tModList *)NULL;
	snprintf(buf, BUFSIZE, "%sdrivers", GetLibDir ());
	GfModInfoDir(CAR_IDENT, buf, 1, &list);

	curmod = list;
	if (curmod != NULL) {
		do {
			curmod = curmod->next;
			for (i = 0; i < MAX_MOD_ITF; i++) {
				if (curmod->modInfo[i].name) {
					sp = strrchr(curmod->sopath, '/');
					if (sp == NULL) {
						sp = curmod->sopath;
					} else {
						sp++;
					}
					strcpy(dname, sp);
					dname[strlen(dname) - strlen(DLLEXT) - 1] = 0; /* cut .so or .dll */
					snprintf(buf, BUFSIZE, "%sdrivers/%s/%s.xml", GetLocalDir(), dname, dname);
					robhdle = GfParmReadFile(buf, GFPARM_RMODE_STD);
					if (!robhdle) {
						snprintf(buf, BUFSIZE, "drivers/%s/%s.xml", dname, dname);
						robhdle = GfParmReadFile(buf, GFPARM_RMODE_STD);
					}
					snprintf(path, BUFSIZE, "%s/%s/%d", ROB_SECT_ROBOTS, ROB_LIST_INDEX, curmod->modInfo[i].index);
					printf("Robot Name=%s, path=%s\n",dname,path);
					const char* carName = GfParmGetStr(robhdle, path, ROB_ATTR_CAR, "");
					if (strcmp(GfParmGetStr(robhdle, path, ROB_ATTR_TYPE, ROB_VAL_ROBOT), ROB_VAL_ROBOT)) {
						human = 1;
					} else {
						human = 0;
					}
					snprintf(path, BUFSIZE, "cars/%s/%s.xml", carName, carName);
					if (!stat(path, &st)) {
						carhdle = GfParmReadFile(path, GFPARM_RMODE_STD);
						if (carhdle) {
							curDrv = (tDrvElt*)calloc(1, sizeof(tDrvElt));
							curDrv->index = curmod->modInfo[i].index;
							curDrv->dname = strdup(dname);
							curDrv->name = strdup(curmod->modInfo[i].name);
							curDrv->car = carhdle;
							if (human) {
								curDrv->human = 1;
								GF_TAILQ_INSERT_HEAD(&DrvList, curDrv, link);
							} else {
								curDrv->human = 0;
								GF_TAILQ_INSERT_TAIL(&DrvList, curDrv, link);
							}
						} else {
							GfOut("Driver %s not selected because car %s is not readable\n", curmod->modInfo[i].name, carName);
						}
					} else {
						GfOut("Driver %s not selected because car %s is not present\n", curmod->modInfo[i].name, carName);
					}
					GfParmReleaseHandle(robhdle);
				}
			}
		} while (curmod != list);
		
		GfModFreeInfoList(&list);
	}
	
	nbSelectedDrivers = 0;
	nbMaxSelectedDrivers = (int)GfParmGetNum(ds->param, RM_SECT_DRIVERS, RM_ATTR_MAXNUM, NULL, 0);
	nCars = GfParmGetEltNb(ds->param, RM_SECT_DRIVERS);
	index = 1;
	for (i = 1; i < nCars+1; i++) {
		snprintf(dname, BUFSIZE, "%s/%d", RM_SECT_DRIVERS, i);
		const char* cardllname = GfParmGetStr(ds->param, dname, RM_ATTR_MODULE, "");
		robotIdx = (int)GfParmGetNum(ds->param, dname, RM_ATTR_IDX, (char*)NULL, 0);
	
		curDrv = GF_TAILQ_FIRST(&DrvList);
		if (curDrv != NULL) {
			do {
				if ((curDrv->index == robotIdx) && (strcmp(curDrv->dname, cardllname) == 0)) {
					if (nbSelectedDrivers < nbMaxSelectedDrivers) {
						GfuiScrollListInsertElement(scrHandle, selectedScrollList, curDrv->name, index, (void*)curDrv);
						curDrv->sel = index++;
						nbSelectedDrivers++;
					}
					break;
				}
			} while ((curDrv = GF_TAILQ_NEXT(curDrv, link)) != NULL);
		}
	}

	curDrv = GF_TAILQ_FIRST(&DrvList);
	if (curDrv != NULL) {
		do {
			if (curDrv->sel == 0) {
			GfuiScrollListInsertElement(scrHandle, unselectedScrollList, curDrv->name, 1000, (void*)curDrv);
			}
		} while ((curDrv = GF_TAILQ_NEXT(curDrv, link)) != NULL);
	}

	GfuiLabelCreate(scrHandle, "Focused:", GFUI_FONT_MEDIUM, 320, B_BASE - 5 * B_HT, GFUI_ALIGN_HC_VB, 0);
	const char* cardllname = GfParmGetStr(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSED, "");
	robotIdx = (int)GfParmGetNum(ds->param, RM_SECT_DRIVERS, RM_ATTR_FOCUSEDIDX, (char*)NULL, 0);
	curDrv = GF_TAILQ_FIRST(&DrvList);
	if (curDrv != NULL) {
		do {
			if ((curDrv->index == robotIdx) && (strcmp(curDrv->dname, cardllname) == 0)) {
			break;
			}
		} while ((curDrv = GF_TAILQ_NEXT(curDrv, link)) != NULL);
	}

	if (curDrv == NULL) {
		curDrv = GF_TAILQ_FIRST(&DrvList);
	}
 
	if (curDrv == NULL) {
		FocDrvLabelId = GfuiLabelCreate(scrHandle, "", GFUI_FONT_MEDIUM_C,
						320, B_BASE - 5 * B_HT - GfuiFontHeight(GFUI_FONT_MEDIUM), GFUI_ALIGN_HC_VB, 256);
    } else {
		FocDrvLabelId = GfuiLabelCreate(scrHandle, curDrv->name, GFUI_FONT_MEDIUM_C,
						320, B_BASE - 5 * B_HT - GfuiFontHeight(GFUI_FONT_MEDIUM), GFUI_ALIGN_HC_VB, 256);
    }

	/* Picked Driver Info */
	GfuiLabelCreate(scrHandle, "Driver:", GFUI_FONT_MEDIUM, 320, B_BASE - 7 * B_HT, GFUI_ALIGN_HC_VB, 0);
	PickDrvNameLabelId = GfuiLabelCreateEx(scrHandle, "", aColor, GFUI_FONT_MEDIUM_C,
						320, B_BASE - 7 * B_HT - GfuiFontHeight(GFUI_FONT_MEDIUM), GFUI_ALIGN_HC_VB, 256);
	GfuiLabelCreate(scrHandle, "Car:", GFUI_FONT_MEDIUM, 320, B_BASE - 8 * B_HT, GFUI_ALIGN_HC_VB, 0);
	PickDrvCarLabelId = GfuiLabelCreateEx(scrHandle, "", aColor, GFUI_FONT_MEDIUM_C,
						320, B_BASE - 8 * B_HT - GfuiFontHeight(GFUI_FONT_MEDIUM), GFUI_ALIGN_HC_VB, 256);
	GfuiLabelCreate(scrHandle, "Category:", GFUI_FONT_MEDIUM, 320, B_BASE - 9 * B_HT, GFUI_ALIGN_HC_VB, 0);
	PickDrvCategoryLabelId = GfuiLabelCreateEx(scrHandle, "", aColor, GFUI_FONT_MEDIUM_C,
							320, B_BASE - 9 * B_HT - GfuiFontHeight(GFUI_FONT_MEDIUM), GFUI_ALIGN_HC_VB, 256);
	GfuiMenuDefaultKeysAdd(scrHandle);
	rmdsAddKeys();
	
	GfuiScreenActivate(scrHandle);
}


static void rmFreeDrvList(void)
{
	tDrvElt	*cur;

	while ((cur = GF_TAILQ_FIRST(&DrvList)) != NULL) {
		GF_TAILQ_REMOVE(&DrvList, cur, link);
		free(cur->name);
		free(cur->dname);
		GfParmReleaseHandle(cur->car);
		free(cur);
	}
}
