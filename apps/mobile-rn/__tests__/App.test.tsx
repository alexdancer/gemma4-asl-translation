/**
 * @format
 */


jest.mock('@react-native-documents/picker', () => ({
  pick: jest.fn(),
  isErrorWithCode: jest.fn(() => false),
  types: {video: 'public.movie'},
}));

import App from '../App';

test('exports cloud translation app component', () => {
  expect(App).toBeDefined();
});
