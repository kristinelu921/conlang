/** Check replay interaction against an already served recording; never run models. */
const assert = require('node:assert/strict');
const {chromium} = require('playwright');

/** @returns {Promise<void>} Verify seeking, playback, inspection, races, and a narrow viewport. */
async function main() {
    const address = process.argv[2] ?? 'http://127.0.0.1:8769';
    const timeline = await (await fetch(`${address}/api/timeline`)).json();
    assert.ok(timeline.turns.length >= 3, 'Use a recording with at least three frames');
    const browser = await chromium.launch({headless: true, args: ['--disable-gpu']});
    try {
        const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(address);
        await page.waitForFunction(turn => document.getElementById('turn').textContent === String(turn), timeline.turns[0]);
        await page.waitForSelector('#item-guide table');
        const rules = await (await fetch(`${address}/api/rules`)).json();
        const guideRules = rules.recorded?.item_utility ? rules.recorded : rules.defaults;
        assert.equal(await page.locator('#item-guide tr').count(), Object.keys(guideRules.items).length + 1);
        assert.ok((await page.locator('#item-guide').textContent()).includes('production speed'));
        if (rules.recorded && !rules.recorded.item_utility) {
            assert.ok((await page.locator('#rules-note').textContent()).includes('older recording'));
        }
        assert.equal(await page.locator('#timeline').inputValue(), '0');
        const firstSnapshot = await (await fetch(`${address}/api/snapshot?turn=${timeline.turns[0]}`)).json();
        assert.equal(await page.locator('#population').textContent(), String(firstSnapshot.metrics.population));
        if (firstSnapshot.islands.length) {
            await page.locator('#map .island').first().click();
            assert.ok((await page.locator('#details').textContent()).includes('Inventory'));
        }
        if (firstSnapshot.islands.length > 20) {
            for (const index of [0, Math.floor(firstSnapshot.islands.length / 2), firstSnapshot.islands.length - 1]) {
                const island = firstSnapshot.islands[index];
                await page.locator('#map .island').nth(index).click();
                assert.equal(await page.locator('#selection').inputValue(), String(island.id));
                assert.ok((await page.locator('.map-head p').textContent()).includes(`Routes and cargo for island ${island.id}`));
                const channels = firstSnapshot.channels.filter(channel => channel.source === island.id || channel.target === island.id);
                assert.equal(await page.locator('#map line').count(), channels.length);
            }
            await page.locator('#selection').selectOption(String(firstSnapshot.islands[0].id));
        }
        await page.locator('#next').click();
        await page.waitForFunction(turn => document.getElementById('turn').textContent === String(turn), timeline.turns[1]);
        assert.ok((await page.locator('#details').textContent()).includes('Choice this turn'));
        if (rules.recorded?.schema_version >= 8) {
            const detail = await page.locator('#details').textContent();
            assert.ok(detail.includes('Communication neighbors'));
            assert.ok(detail.includes('Grain and water producers are both in this neighborhood.'));
            assert.ok(detail.includes('Recent private experience'));
            assert.ok(detail.includes(`Turn ${timeline.turns[1]}:`));
        }
        const scoreResponse = await fetch(`${address}/api/decision?turn=${timeline.turns[1]}&island=${firstSnapshot.islands[0].id}`);
        if (scoreResponse.ok) {
            const scores = await scoreResponse.json();
            await page.waitForSelector('#policy-options tbody tr');
            assert.equal(await page.locator('#policy-options tbody tr').count(), scores.stages.action.options.length);
            assert.equal(await page.locator('#policy-options .policy-chosen').count(), 1);
            assert.ok((await page.locator('#policy-summary').textContent()).includes(`${scores.stages.action.options.length} options`));
            assert.ok((await page.locator('#policy-explanation').textContent()).includes('softmax'));
            await page.locator('#policy-filter').fill('no_matching_action_123');
            assert.equal(await page.locator('#policy-options tbody tr').count(), 0);
            await page.locator('#policy-filter').fill('');
            if (scores.stages.quantity) {
                await page.locator('#policy-stage').selectOption('quantity');
                assert.equal(await page.locator('#policy-options tbody tr').count(), scores.stages.quantity.options.length);
                assert.equal(await page.locator('#policy-options .policy-chosen').count(), 1);
                await page.locator('#policy-stage').selectOption('action');
            }
            assert.equal(await page.locator('#policy-download').getAttribute('href'), `api/decision?turn=${timeline.turns[1]}&island=${firstSnapshot.islands[0].id}`);
        }
        const middle = Math.floor(timeline.turns.length / 2);
        await page.locator('#timeline').fill(String(middle));
        await page.waitForFunction(turn => document.getElementById('turn').textContent === String(turn), timeline.turns[middle]);
        await page.waitForTimeout(2200);
        assert.equal(await page.locator('#turn').textContent(), String(timeline.turns[middle]), 'Background refresh must not move a paused playhead');
        await page.locator('#speed').selectOption('4');
        await page.locator('#play').click();
        await page.waitForFunction(turn => Number(document.getElementById('turn').textContent) > turn, timeline.turns[middle]);
        await page.locator('#play').click();
        const pausedTurn = await page.locator('#turn').textContent();
        await page.waitForTimeout(700);
        assert.equal(await page.locator('#turn').textContent(), pausedTurn);
        await page.locator('#last').click();
        await page.waitForFunction(turn => document.getElementById('turn').textContent === String(turn), timeline.turns.at(-1));
        const finalSnapshot = await (await fetch(`${address}/api/snapshot`)).json();
        assert.equal(await page.locator('#population').textContent(), String(finalSnapshot.metrics.population));
        if (finalSnapshot.deceased?.length) {
            await page.locator('#selection').selectOption(String(finalSnapshot.deceased[0].id));
            assert.ok((await page.locator('#details').textContent()).includes('Died at turn'));
        }
        await page.locator('#restart').click();
        await page.waitForFunction(turn => document.getElementById('turn').textContent === String(turn), timeline.turns[0]);
        // A slow old request must not replace a newer seek.
        await page.reload();
        await page.waitForFunction(() => !document.getElementById('play').disabled);
        await page.route(`**/api/snapshot?turn=${timeline.turns[1]}`, async route => {
            await new Promise(resolve => setTimeout(resolve, 400));
            await route.continue();
        });
        await page.locator('#timeline').fill('1');
        await page.locator('#timeline').fill('2');
        await page.waitForFunction(turn => document.getElementById('turn').textContent === String(turn), timeline.turns[2]);
        await page.waitForTimeout(600);
        assert.equal(await page.locator('#turn').textContent(), String(timeline.turns[2]));
        await page.unrouteAll({behavior: 'wait'});
        await page.setViewportSize({width: 390, height: 844});
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), 'Page must fit a phone viewport');
        assert.deepEqual(errors, []);
        if (process.env.ISLAND_REPLAY_SCREENSHOT) {
            await page.setViewportSize({width: 1440, height: 1100});
            await page.screenshot({path: process.env.ISLAND_REPLAY_SCREENSHOT, fullPage: true});
        }
        // A labeled UI fixture exercises new inspector fields without running a simulation.
        const fixture = structuredClone(firstSnapshot);
        fixture.islands[0].food_reserve = 1;
        fixture.islands[0].active_effects = {tool: fixture.turn + 9};
        fixture.islands[0].item_bonuses = {productivity: 0.5, recovery: 0, comfort: 0, disease_resistance: 0, loss_protection: 0};
        fixture.maximum_health = 300;
        fixture.islands[0].health = 293;
        fixture.decisions = [{island_id: fixture.islands[0].id, action: {kind: 'wait', args: {}}, reward: -8,
            terminal: false, reward_components: {resource_reward: 0.5, empty_food_penalty: -5,
                health_loss_penalty: -3.5, health_lost: 7, food_available: 0, total: -8}}];
        await page.route('**/api/snapshot?turn=*', route => route.fulfill({json: fixture}));
        await page.reload();
        await page.waitForFunction(() => document.getElementById('details').textContent.includes('Saved nutrition'));
        assert.ok((await page.locator('#details').textContent()).includes('+50% production speed'));
        assert.ok((await page.locator('#details').textContent()).includes(`through turn ${fixture.turn + 9}`));
        const details = await page.locator('#details').textContent();
        assert.ok(details.includes('293 / 300'));
        assert.ok(details.includes('Learning reward: -8'));
        assert.ok(details.includes('Empty food penalty: -5'));
        assert.ok(details.includes('Health loss penalty: -3.5'));
        assert.ok(details.includes('Health lost this turn: 7'));
        assert.ok(!details.includes('Empty water penalty'), 'Older recordings must retain their original breakdown');
        fixture.decisions[0].reward = -13;
        Object.assign(fixture.decisions[0].reward_components,
            {water_available: 0, empty_water_penalty: -5, total: -13});
        await page.reload();
        await page.waitForFunction(() => document.getElementById('details').textContent.includes('Empty water penalty'));
        const waterDetails = await page.locator('#details').textContent();
        assert.ok(waterDetails.includes('Learning reward: -13'));
        assert.ok(waterDetails.includes('Empty food penalty: -5'));
        assert.ok(waterDetails.includes('Empty water penalty: -5'));
        assert.ok(waterDetails.includes('Usable water remaining: 0'));
        assert.deepEqual(errors, []);
        console.log(`Replay browser checks passed for ${timeline.turns.length} frames.`);
    } finally {
        await browser.close();
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
